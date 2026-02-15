"""
Build PREFLOP V1 structured metadata features from EDGAR + existing daily bars.

Features built:
  Tier 1 (from existing data, no API calls):
    - preflop_s1_lead_days:     first_bar_date - earliest_s1_date (filing lead time)
    - preflop_exchange_nyse:    1.0 if NYSE, else 0.0
    - preflop_exchange_nasdaq:  1.0 if Nasdaq, else 0.0

  Tier 2 (from EDGAR S-1/A query):
    - preflop_s1a_count:               number of S-1/A amendments per CIK
    - preflop_days_since_last_amendment: first_bar_date - last_amendment_date

Output:
    data/features/preflop_metadata.parquet

Usage:
    python -m scripts.build_preflop_metadata
    python -m scripts.build_preflop_metadata --skip-edgar   # Tier 1 only (offline)
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

import polars as pl

EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
USER_AGENT = "ipo-risk-engine/1.0 (learning-project@example.com)"
RAW_DIR = Path("data/raw")
UNIVERSE_PATH = Path("data/ipo_universe_alpaca_v1.parquet")
OUTPUT_PATH = Path("data/features/preflop_metadata.parquet")


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def compute_first_bar_dates(symbols: list[str]) -> dict[str, date]:
    """Get the earliest daily bar date per symbol from cached bars."""
    first_dates: dict[str, date] = {}
    for sym in symbols:
        bars_path = RAW_DIR / sym / "bars_1d.parquet"
        if not bars_path.exists():
            continue
        try:
            bars = pl.read_parquet(bars_path)
            if bars.height > 0:
                first_ts = bars.sort("ts")["ts"][0]
                first_dates[sym] = first_ts.date()  # type: ignore[union-attr]
        except Exception:
            continue
    return first_dates


def fetch_s1a_filings(
    start_date: str,
    end_date: str,
    verbose: bool = False,
) -> list[dict]:
    """Fetch all S-1/A amendment filings from EDGAR EFTS."""
    filings: list[dict] = []
    offset = 0
    page_size = 100

    while True:
        params = urllib.parse.urlencode({
            "forms": "S-1/A",
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
            print(f"  EDGAR HTTP {exc.code} at offset={offset}, stopping.")
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

            filings.append({
                "cik": ciks[0].lstrip("0") or "0",
                "file_date": file_date,
            })

        if verbose:
            print(f"  [EFTS S-1/A] offset={offset} got={len(hits)} total={total}")

        offset += page_size
        if offset >= total or not hits:
            break
        time.sleep(0.12)

    return filings


def group_amendments_by_cik(
    filings: list[dict],
) -> dict[str, dict]:
    """Group S-1/A filings by CIK -> {count, last_date}."""
    by_cik: dict[str, list[date]] = {}
    for f in filings:
        by_cik.setdefault(f["cik"], []).append(f["file_date"])

    result: dict[str, dict] = {}
    for cik, dates in by_cik.items():
        result[cik] = {
            "s1a_count": len(dates),
            "last_amendment_date": max(dates),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PREFLOP V1 metadata")
    parser.add_argument("--skip-edgar", action="store_true",
                        help="Skip S-1/A EDGAR query (Tier 1 only)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("BUILD PREFLOP V1 METADATA")
    print("=" * 60)

    univ = pl.read_parquet(UNIVERSE_PATH)
    symbols = univ["symbol"].to_list()
    print(f"  Universe: {len(symbols)} symbols")

    cik_map = {
        row["symbol"]: row["cik"]
        for row in univ.iter_rows(named=True)
        if row["cik"]
    }

    print("\n  Computing first bar dates...")
    first_bars = compute_first_bar_dates(symbols)
    print(f"  Found daily bars for {len(first_bars)}/{len(symbols)} symbols")

    rows: list[dict] = []
    for row in univ.iter_rows(named=True):
        sym = row["symbol"]
        s1_date = row["ipo_date"]  # Earliest S-1 filing date
        exchange = (row["exchange"] or "").lower()

        first_bar_date = first_bars.get(sym)
        if first_bar_date is not None:
            s1_lead_days = (first_bar_date - s1_date).days
        else:
            s1_lead_days = None

        rows.append({
            "symbol": sym,
            "s1_filing_date": s1_date,
            "first_bar_date": first_bar_date,
            "preflop_s1_lead_days": float(s1_lead_days) if s1_lead_days is not None else None,
            "preflop_exchange_nyse": 1.0 if "nyse" in exchange else 0.0,
            "preflop_exchange_nasdaq": 1.0 if "nasdaq" in exchange else 0.0,
            "preflop_s1a_count": None,
            "preflop_days_since_last_amendment": None,
        })

    if not args.skip_edgar:
        ipo_min = univ["ipo_date"].min()
        ipo_max = univ["ipo_date"].max()
        s1a_filings: list[dict] = []
        min_year = ipo_min.year if ipo_min else 2015  # type: ignore[union-attr]
        max_year = ipo_max.year if ipo_max else 2025  # type: ignore[union-attr]
        for yr in range(min_year, max_year + 1):
            yr_start = f"{yr}-01-01"
            yr_end = f"{yr}-12-31"
            print(f"\n  Fetching S-1/A amendments for {yr}...")
            yr_filings = fetch_s1a_filings(yr_start, yr_end, verbose=args.verbose)
            s1a_filings.extend(yr_filings)
            print(f"    -> {len(yr_filings)} filings")
        print(f"\n  Total S-1/A filings: {len(s1a_filings)}")

        amendments = group_amendments_by_cik(s1a_filings)
        print(f"  Unique CIKs with amendments: {len(amendments)}")

        matched = 0
        for r in rows:
            sym = r["symbol"]
            cik = cik_map.get(sym)
            if cik and cik in amendments:
                amend = amendments[cik]
                r["preflop_s1a_count"] = float(amend["s1a_count"])
                first_bar = r["first_bar_date"]
                if first_bar is not None:
                    days_since = (first_bar - amend["last_amendment_date"]).days
                    r["preflop_days_since_last_amendment"] = float(days_since)
                matched += 1
            else:
                r["preflop_s1a_count"] = 0.0
                r["preflop_days_since_last_amendment"] = None

        print(f"  Matched amendments: {matched}/{len(rows)} symbols")
    else:
        print("\n  Skipping EDGAR S-1/A query (--skip-edgar)")

    df = pl.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT_PATH)

    print(f"\n  Saved: {OUTPUT_PATH} ({df.height} rows, {len(df.columns)} cols)")
    print(f"  Columns: {df.columns}")

    valid = df.filter(pl.col("preflop_s1_lead_days").is_not_null())
    print(f"\n  S1 lead days: median={valid['preflop_s1_lead_days'].median():.0f}, "
          f"mean={valid['preflop_s1_lead_days'].mean():.0f}, "
          f"range=[{valid['preflop_s1_lead_days'].min():.0f}, "
          f"{valid['preflop_s1_lead_days'].max():.0f}]")

    if not args.skip_edgar:
        s1a_valid = df.filter(pl.col("preflop_s1a_count").is_not_null())
        print(f"  S1/A count: median={s1a_valid['preflop_s1a_count'].median():.0f}, "
              f"mean={s1a_valid['preflop_s1a_count'].mean():.1f}, "
              f"range=[{s1a_valid['preflop_s1a_count'].min():.0f}, "
              f"{s1a_valid['preflop_s1a_count'].max():.0f}]")


if __name__ == "__main__":
    main()
