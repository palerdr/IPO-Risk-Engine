"""
Build a historical IPO universe from SEC EDGAR filings.

Strategy:
  1. Fetch all S-1 and F-1 filings from EDGAR EFTS (initial IPO registrations)
  2. Group by CIK, take earliest filing date per company -> approximate IPO date
  3. Resolve CIK -> ticker + exchange via company_tickers_exchange.json
  4. For unresolved CIKs, extract ticker from EDGAR display_name field
  5. Apply deterministic filters (SPAC exclusion, US exchange, common equity)
  6. Optionally validate first-trade date via Alpaca

Output:
  data/ipo_universe.parquet          — included universe
  data/ipo_universe_audit.parquet    — all rows with exclusion reasons

Usage:
  python scripts/build_universe_edgar.py
  python scripts/build_universe_edgar.py --start-date 2015-01-01 --end-date 2025-12-31
  python scripts/build_universe_edgar.py --validate-alpaca --validate-limit 50
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
USER_AGENT = "ipo-risk-engine/1.0 (learning-project@example.com)"
DEFAULT_OUTPUT = Path("data/ipo_universe.parquet")

SPAC_KEYWORDS = (
    "acquisition corp", "acquisition corporation", "blank check", "spac",
    "capital corp", "merger corp", "holdings acquisition",
)
EXCLUDED_NAME_TERMS = (
    " etf", " fund", " trust", " warrant", " unit", " rights",
    " preferred", " notes", " adr",
)
ALLOWED_EXCHANGES = ("nasdaq", "nyse", "nyse american", "nyse arca")

# Regex to extract ticker from EDGAR display_name: "Company Name  (TICK)  (CIK ...)"
_TICKER_RE = re.compile(r"\(([A-Z]{1,5})\)")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EdgarFiling:
    cik: str
    company_name: str
    display_name: str
    file_date: date
    form_type: str
    sic: str | None
    state: str | None


@dataclass
class UniverseRow:
    symbol: str
    company_name: str
    ipo_date: date  # earliest S-1/F-1 filing date
    cik: str
    exchange: str | None
    sic: str | None
    state: str | None
    ticker_source: str  # "company_tickers" or "display_name" or "unknown"
    filing_count: int


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build IPO universe from SEC EDGAR")
    p.add_argument("--start-date", default="2010-01-01")
    p.add_argument("--end-date", default=date.today().isoformat())
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--validate-alpaca", action="store_true")
    p.add_argument("--validate-limit", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# EDGAR EFTS fetch
# ---------------------------------------------------------------------------
def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_filings(
    forms: str,
    start_date: str,
    end_date: str,
    verbose: bool = False,
) -> list[EdgarFiling]:
    """Paginate through EDGAR EFTS for given form types."""
    filings: list[EdgarFiling] = []
    offset = 0
    page_size = 100

    while True:
        params = urllib.parse.urlencode({
            "forms": forms,
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "from": offset,
            "size": page_size,
        })
        url = f"{EFTS_URL}?{params}"

        try:
            data = _fetch_json(url)
        except urllib.error.HTTPError as exc:
            print(f"  EDGAR HTTP {exc.code} at offset={offset}, stopping pagination.")
            break

        hits = data.get("hits", {}).get("hits", [])
        total = data.get("hits", {}).get("total", {}).get("value", 0)

        for hit in hits:
            src = hit.get("_source", {})
            ciks = src.get("ciks", [])
            if not ciks:
                continue

            file_date_str = src.get("file_date", "")
            try:
                file_date = datetime.strptime(file_date_str[:10], "%Y-%m-%d").date()
            except ValueError:
                continue

            display_names = src.get("display_names", [])
            display_name = display_names[0] if display_names else ""
            # Extract entity name (part before the first parenthesis)
            entity_name = display_name.split("(")[0].strip() if display_name else ""

            filings.append(EdgarFiling(
                cik=ciks[0].lstrip("0") or "0",
                company_name=entity_name,
                display_name=display_name,
                file_date=file_date,
                form_type=src.get("form", ""),
                sic=src.get("sics", [None])[0] if src.get("sics") else None,
                state=src.get("biz_states", [None])[0] if src.get("biz_states") else None,
            ))

        if verbose:
            print(f"  [EFTS] offset={offset} got={len(hits)} total={total}")

        offset += page_size
        if offset >= total or not hits:
            break

        # Respect SEC rate limit: 10 req/sec
        time.sleep(0.12)

    return filings


# ---------------------------------------------------------------------------
# CIK -> ticker resolution
# ---------------------------------------------------------------------------
def fetch_ticker_map() -> dict[str, dict]:
    """Download company_tickers_exchange.json -> {cik_str: {ticker, name, exchange}}."""
    print("  Downloading company_tickers_exchange.json ...")
    data = _fetch_json(TICKERS_URL)
    fields = data["fields"]  # ['cik', 'name', 'ticker', 'exchange']
    rows = data["data"]

    ticker_map: dict[str, dict] = {}
    for row in rows:
        entry = dict(zip(fields, row))
        cik_str = str(entry["cik"])
        ticker_map[cik_str] = {
            "ticker": entry.get("ticker", ""),
            "name": entry.get("name", ""),
            "exchange": entry.get("exchange", ""),
        }
    print(f"  Loaded {len(ticker_map)} CIK->ticker mappings")
    return ticker_map


def _extract_ticker_from_display(display_name: str) -> str | None:
    """Try to extract ticker from 'Company Name  (TICK)  (CIK ...)'."""
    matches = _TICKER_RE.findall(display_name)
    # Filter out 'CIK' and pure numbers
    for m in matches:
        if m != "CIK" and not m.isdigit():
            return m
    return None


# ---------------------------------------------------------------------------
# Group filings -> universe rows
# ---------------------------------------------------------------------------
def build_universe_rows(
    filings: list[EdgarFiling],
    ticker_map: dict[str, dict],
) -> list[UniverseRow]:
    """Group filings by CIK, resolve tickers, build universe rows."""
    # Group by CIK: keep earliest filing date, accumulate metadata
    by_cik: dict[str, list[EdgarFiling]] = {}
    for f in filings:
        by_cik.setdefault(f.cik, []).append(f)

    rows: list[UniverseRow] = []
    resolved = 0
    display_resolved = 0
    unresolved = 0

    for cik, group in by_cik.items():
        earliest = min(group, key=lambda f: f.file_date)

        # Resolve ticker
        mapped = ticker_map.get(cik)
        if mapped and mapped["ticker"]:
            symbol = mapped["ticker"]
            exchange = mapped["exchange"]
            ticker_source = "company_tickers"
            resolved += 1
        else:
            # Try extracting from display_name
            extracted = _extract_ticker_from_display(earliest.display_name)
            if extracted:
                symbol = extracted
                exchange = None
                ticker_source = "display_name"
                display_resolved += 1
            else:
                symbol = f"CIK:{cik}"
                exchange = None
                ticker_source = "unknown"
                unresolved += 1

        rows.append(UniverseRow(
            symbol=symbol,
            company_name=earliest.company_name or (mapped["name"] if mapped else ""),
            ipo_date=earliest.file_date,
            cik=cik,
            exchange=exchange,
            sic=earliest.sic,
            state=earliest.state,
            ticker_source=ticker_source,
            filing_count=len(group),
        ))

    print(f"  Ticker resolution: {resolved} from tickers file, "
          f"{display_resolved} from display_name, {unresolved} unresolved")
    return sorted(rows, key=lambda r: (r.ipo_date, r.symbol))


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def _normalize(s: str | None) -> str:
    return (s or "").strip().lower()


def apply_filters(rows: list[UniverseRow]) -> tuple[list[dict], list[dict]]:
    """Apply SPAC/exchange/equity filters. Returns (kept, all_with_reasons)."""
    kept, all_rows = [], []

    for row in rows:
        reasons: list[str] = []
        name_lower = _normalize(row.company_name)
        exchange_lower = _normalize(row.exchange)

        # SPAC filter
        if any(kw in name_lower for kw in SPAC_KEYWORDS):
            reasons.append("spac_name")

        # Exchange filter (allow if exchange is known and matches)
        if row.exchange and not any(ex in exchange_lower for ex in ALLOWED_EXCHANGES):
            reasons.append("non_us_exchange")
        elif not row.exchange and row.ticker_source == "unknown":
            reasons.append("no_ticker_or_exchange")

        # Security type filter by name
        if any(term in name_lower for term in EXCLUDED_NAME_TERMS):
            reasons.append("non_common_by_name")

        # Skip CIK-only symbols (no usable ticker)
        if row.symbol.startswith("CIK:"):
            reasons.append("no_ticker_resolved")

        # Ticker-pattern filter: warrants (W suffix), units (U suffix), preferreds (-)
        sym = row.symbol.upper()
        if len(sym) > 3 and sym.endswith("W"):
            reasons.append("warrant_by_ticker")
        if len(sym) > 3 and sym.endswith("U"):
            reasons.append("unit_by_ticker")
        if "-" in sym:
            reasons.append("preferred_by_ticker")

        row_dict = {
            "symbol": row.symbol,
            "company_name": row.company_name,
            "ipo_date": row.ipo_date,
            "cik": row.cik,
            "exchange": row.exchange,
            "sic": row.sic,
            "state": row.state,
            "ticker_source": row.ticker_source,
            "filing_count": row.filing_count,
            "excluded_reasons": ",".join(reasons),
            "included": len(reasons) == 0,
        }
        all_rows.append(row_dict)
        if not reasons:
            kept.append(row_dict)

    return kept, all_rows


# ---------------------------------------------------------------------------
# Optional Alpaca validation (reuse from build_ipo_universe.py)
# ---------------------------------------------------------------------------
def alpaca_validate(rows: list[dict], limit: int) -> list[dict]:
    """Add first-trade date and bar count from Alpaca."""
    from datetime import timedelta

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    from ipo_risk_engine.config.settings import load_settings

    settings = load_settings()
    client = StockHistoricalDataClient(
        api_key=settings.api_key, secret_key=settings.api_secret,
    )

    max_rows = len(rows) if limit <= 0 else min(limit, len(rows))
    out: list[dict] = []

    for idx, row in enumerate(rows):
        validated = dict(row)

        if idx < max_rows:
            ipo_d = row["ipo_date"]
            start = datetime.combine(
                ipo_d - timedelta(days=3), datetime.min.time(), tzinfo=timezone.utc,
            )
            end = datetime.combine(
                ipo_d + timedelta(days=30), datetime.min.time(), tzinfo=timezone.utc,
            )

            try:
                bars = client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=row["symbol"],
                    timeframe=TimeFrame(1, TimeFrameUnit.Day),
                    start=start, end=end,
                ))
                bar_list = bars.data.get(row["symbol"], [])

                if bar_list:
                    first_ts = min(x.timestamp for x in bar_list)
                    validated["alpaca_first_trade_date"] = first_ts.date()
                    validated["alpaca_bar_count_30d"] = len(bar_list)
                else:
                    validated["alpaca_first_trade_date"] = None
                    validated["alpaca_bar_count_30d"] = 0
            except Exception:
                validated["alpaca_first_trade_date"] = None
                validated["alpaca_bar_count_30d"] = 0

        out.append(validated)

        if (idx + 1) % 50 == 0 or idx + 1 == len(rows):
            print(f"  [alpaca] validated {idx + 1}/{len(rows)}")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    print(f"Building IPO universe from SEC EDGAR ({args.start_date} to {args.end_date})")

    # 1. Fetch CIK->ticker mapping
    ticker_map = fetch_ticker_map()

    # 2. Fetch S-1 and F-1 filings
    print(f"\n  Fetching S-1 filings ...")
    s1_filings = fetch_filings("S-1", args.start_date, args.end_date, args.verbose)
    print(f"  Got {len(s1_filings)} S-1 filings")

    print(f"  Fetching F-1 filings ...")
    f1_filings = fetch_filings("F-1", args.start_date, args.end_date, args.verbose)
    print(f"  Got {len(f1_filings)} F-1 filings")

    all_filings = s1_filings + f1_filings
    print(f"\n  Total filings: {len(all_filings)}")

    # 3. Group by CIK -> universe rows
    universe_rows = build_universe_rows(all_filings, ticker_map)
    print(f"  Unique companies (by CIK): {len(universe_rows)}")

    # 4. Apply filters
    kept, all_rows = apply_filters(universe_rows)
    print(f"  After filters: {len(kept)} included, {len(all_rows) - len(kept)} excluded")

    # 5. Optional Alpaca validation
    if args.validate_alpaca:
        print("\n  Running Alpaca first-trade validation ...")
        kept = alpaca_validate(kept, args.validate_limit)

    # 6. Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df_keep = pl.DataFrame(kept).sort(["ipo_date", "symbol"]) if kept else pl.DataFrame()
    df_keep.write_parquet(args.output)
    print(f"\n  Wrote: {args.output} ({df_keep.height} rows)")

    audit_path = args.output.with_name("ipo_universe_audit.parquet")
    df_all = pl.DataFrame(all_rows).sort(["ipo_date", "symbol"]) if all_rows else pl.DataFrame()
    df_all.write_parquet(audit_path)
    print(f"  Wrote: {audit_path} ({df_all.height} rows)")

    # 7. Summary
    print(f"\n{'='*60}")
    print("UNIVERSE SUMMARY")
    print(f"{'='*60}")
    print(f"  Date range:     {args.start_date} to {args.end_date}")
    print(f"  Total filings:  {len(all_filings)}")
    print(f"  Unique CIKs:    {len(universe_rows)}")
    print(f"  Included:       {len(kept)}")

    if kept:
        dates = [r["ipo_date"] for r in kept]
        print(f"  IPO date range: {min(dates)} to {max(dates)}")

    # Exclusion breakdown
    excluded = [r for r in all_rows if not r["included"]]
    if excluded:
        reason_counts: dict[str, int] = {}
        for r in excluded:
            for reason in r["excluded_reasons"].split(","):
                if reason:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
        print("\n  Exclusion reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    # Year breakdown
    if kept:
        print("\n  IPOs per year:")
        year_counts: dict[int, int] = {}
        for r in kept:
            yr = r["ipo_date"].year
            year_counts[yr] = year_counts.get(yr, 0) + 1
        for yr in sorted(year_counts):
            print(f"    {yr}: {year_counts[yr]}")


if __name__ == "__main__":
    main()
