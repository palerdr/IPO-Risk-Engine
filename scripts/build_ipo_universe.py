"""
Build a historical IPO universe from Intrinio and save to Parquet.

Primary source:
  Intrinio v2 company IPOs endpoint: /companies/ipos

Output:
  data/ipo_universe.parquet

Usage examples:
  python scripts/build_ipo_universe.py
  python scripts/build_ipo_universe.py --start-date 2015-01-01 --end-date 2025-12-31
  python scripts/build_ipo_universe.py --validate-alpaca
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl


INTRINIO_IPOS_URL = "https://api-v2.intrinio.com/companies/ipos"
DEFAULT_OUTPUT = Path("data/ipo_universe.parquet")
FILTER_VERSION = "v1_intrinio_us_common_ex_spac"

SPAC_KEYWORDS = (
    "acquisition corp",
    "acquisition corporation",
    "blank check",
    "spac",
)

EXCLUDED_SECURITY_TERMS = (
    "etf",
    "fund",
    "adr",
    "warrant",
    "unit",
    "rights",
    "preferred",
    "trust",
    "notes",
)

ALLOWED_EXCHANGE_HINTS = (
    "nasdaq",
    "new york stock exchange",
    "nyse",
    "nyse american",
)


@dataclass(frozen=True)
class ParsedIPO:
    symbol: str
    company_name: str
    ipo_date: date
    status: str | None
    exchange_raw: str | None
    security_type_raw: str | None
    source_ipo_id: str | None
    offer_price: float | None
    offer_price_low: float | None
    offer_price_high: float | None
    shares_offered: int | None
    deal_size: float | None
    source_status_query: str
    source_payload: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build historical IPO universe from Intrinio")
    parser.add_argument("--start-date", type=str, default="2010-01-01")
    parser.add_argument("--end-date", type=str, default=date.today().isoformat())
    parser.add_argument("--statuses", type=str, default="priced")
    parser.add_argument("--api-key", type=str, default=os.getenv("INTRINIO_API_KEY"))
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=0, help="0 means no limit")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--include-spacs", action="store_true")
    parser.add_argument("--allow-non-us-exchanges", action="store_true")
    parser.add_argument("--allow-non-common", action="store_true")
    parser.add_argument("--validate-alpaca", action="store_true")
    parser.add_argument("--validate-limit", type=int, default=0, help="0 means all rows")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _parse_iso_date(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        # Handles YYYY-MM-DD and timestamps like 2024-03-21T00:00:00Z
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_non_empty(d: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = d.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _extract_company_field(payload: dict[str, Any], keys: list[str]) -> Any:
    company = payload.get("company")
    if isinstance(company, dict):
        value = _first_non_empty(company, keys)
        if value is not None:
            return value
    return _first_non_empty(payload, keys)


def fetch_intrinio_ipos(
    *,
    start_date: str,
    end_date: str,
    statuses: list[str],
    api_key: str,
    page_size: int,
    max_pages: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pages_total = 0

    for status in statuses:
        next_page: str | None = None
        pages_for_status = 0

        while True:
            if max_pages and pages_total >= max_pages:
                return records

            params: dict[str, Any] = {
                "start_date": start_date,
                "end_date": end_date,
                "status": status,
                "page_size": page_size,
                "api_key": api_key,
            }
            if next_page:
                params["next_page"] = next_page

            url = f"{INTRINIO_IPOS_URL}?{urllib.parse.urlencode(params)}"
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Intrinio HTTP error {exc.code} for status={status}. Body: {body[:500]}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"Intrinio request failed: {exc}") from exc

            ipos = payload.get("ipos")
            if not isinstance(ipos, list):
                ipos = payload.get("data", [])
            if not isinstance(ipos, list):
                raise RuntimeError("Unexpected Intrinio response shape: expected list at ipos/data")

            for item in ipos:
                if isinstance(item, dict):
                    item["_source_status_query"] = status
                    records.append(item)

            pages_for_status += 1
            pages_total += 1
            next_page = payload.get("next_page")

            if verbose:
                print(
                    f"[fetch] status={status} page={pages_for_status} rows={len(ipos)} "
                    f"next_page={'yes' if next_page else 'no'}"
                )

            if not next_page:
                break

    return records


def parse_ipo_record(payload: dict[str, Any]) -> ParsedIPO | None:
    symbol = _extract_company_field(payload, ["ticker", "symbol", "stock_symbol"])
    ipo_date = _parse_iso_date(
        _first_non_empty(payload, ["ipo_date", "listing_date", "priced_date", "first_trade_date"])
    )

    if not symbol or ipo_date is None:
        return None

    company_name = _extract_company_field(payload, ["name", "legal_name", "company_name"]) or ""
    exchange_raw = _extract_company_field(
        payload,
        ["stock_exchange", "exchange", "primary_exchange", "stock_exchange_mic"],
    )
    security_type_raw = _extract_company_field(
        payload,
        ["security_type", "issue_type", "stock_type", "security_class"],
    )

    return ParsedIPO(
        symbol=str(symbol).upper().strip(),
        company_name=str(company_name).strip(),
        ipo_date=ipo_date,
        status=_first_non_empty(payload, ["status", "ipo_status"]),
        exchange_raw=None if exchange_raw is None else str(exchange_raw),
        security_type_raw=None if security_type_raw is None else str(security_type_raw),
        source_ipo_id=None if payload.get("id") is None else str(payload.get("id")),
        offer_price=_to_float(_first_non_empty(payload, ["offer_price", "price"])),
        offer_price_low=_to_float(_first_non_empty(payload, ["offer_price_low", "price_low"])),
        offer_price_high=_to_float(_first_non_empty(payload, ["offer_price_high", "price_high"])),
        shares_offered=_to_int(_first_non_empty(payload, ["shares_offered", "share_count"])),
        deal_size=_to_float(_first_non_empty(payload, ["deal_size", "gross_proceeds"])),
        source_status_query=str(payload.get("_source_status_query", "")),
        source_payload=payload,
    )


def _is_spac_name(company_name: str) -> bool:
    name = _normalize_text(company_name)
    return any(keyword in name for keyword in SPAC_KEYWORDS)


def _is_us_exchange(exchange_raw: str | None) -> bool:
    if not exchange_raw:
        return False
    normalized = _normalize_text(exchange_raw)
    return any(hint in normalized for hint in ALLOWED_EXCHANGE_HINTS)


def _is_common_equity(security_type_raw: str | None) -> bool:
    # Missing/unknown type is common in vendor data; allow it and let filters be conservative elsewhere.
    if not security_type_raw:
        return True

    text = _normalize_text(security_type_raw)
    if any(term in text for term in EXCLUDED_SECURITY_TERMS):
        return False

    # Accept clear common equity labels.
    common_markers = ("common", "ordinary", "class a", "class b", "common stock")
    return any(marker in text for marker in common_markers)


def _dedupe_rows(rows: list[ParsedIPO]) -> list[ParsedIPO]:
    # Keep earliest record per (symbol, ipo_date), preferring those with more fields.
    grouped: dict[tuple[str, date], ParsedIPO] = {}
    for row in rows:
        key = (row.symbol, row.ipo_date)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = row
            continue

        score_existing = int(bool(existing.exchange_raw)) + int(bool(existing.security_type_raw))
        score_new = int(bool(row.exchange_raw)) + int(bool(row.security_type_raw))
        if score_new > score_existing:
            grouped[key] = row

    return sorted(grouped.values(), key=lambda r: (r.ipo_date, r.symbol))


def _apply_filters(
    rows: list[ParsedIPO],
    *,
    include_spacs: bool,
    allow_non_us_exchanges: bool,
    allow_non_common: bool,
    start: date,
    end: date,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for row in rows:
        reasons: list[str] = []

        if row.ipo_date < start or row.ipo_date > end:
            reasons.append("outside_date_window")

        if not include_spacs and _is_spac_name(row.company_name):
            reasons.append("spac_name")

        if not allow_non_us_exchanges and not _is_us_exchange(row.exchange_raw):
            reasons.append("non_us_exchange_or_missing")

        if not allow_non_common and not _is_common_equity(row.security_type_raw):
            reasons.append("non_common_security_type")

        row_dict: dict[str, Any] = {
            "symbol": row.symbol,
            "company_name": row.company_name,
            "ipo_date": row.ipo_date,
            "status": row.status,
            "exchange_raw": row.exchange_raw,
            "security_type_raw": row.security_type_raw,
            "offer_price": row.offer_price,
            "offer_price_low": row.offer_price_low,
            "offer_price_high": row.offer_price_high,
            "shares_offered": row.shares_offered,
            "deal_size": row.deal_size,
            "source_ipo_id": row.source_ipo_id,
            "source_status_query": row.source_status_query,
            "excluded_reasons": ",".join(reasons),
            "included": len(reasons) == 0,
        }
        all_rows.append(row_dict)
        if not reasons:
            kept.append(row_dict)

    return kept, all_rows


def _parse_data_feed(value: str):
    from alpaca.data import DataFeed

    mapping = {
        "sip": DataFeed.SIP,
        "iex": DataFeed.IEX,
    }
    return mapping.get(value.strip().lower(), DataFeed.SIP)


def _alpaca_validate_first_trade(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    from ipo_risk_engine.config.settings import load_settings

    settings = load_settings()
    client = StockHistoricalDataClient(api_key=settings.api_key, secret_key=settings.api_secret)
    feed = _parse_data_feed(settings.data_feed)

    out: list[dict[str, Any]] = []
    max_rows = len(rows) if limit <= 0 else min(limit, len(rows))

    for idx, row in enumerate(rows):
        validated = dict(row)

        if idx < max_rows:
            ipo_d = row["ipo_date"]
            start = datetime.combine(ipo_d - timedelta(days=3), datetime.min.time(), tzinfo=timezone.utc)
            end = datetime.combine(ipo_d + timedelta(days=30), datetime.min.time(), tzinfo=timezone.utc)

            try:
                request = StockBarsRequest(
                    symbol_or_symbols=row["symbol"],
                    timeframe=TimeFrame(1, TimeFrameUnit.Day),
                    start=start,
                    end=end,
                    feed=feed,
                )
                bars = client.get_stock_bars(request)
                bar_list = bars.data.get(row["symbol"], [])

                if bar_list:
                    first_ts = min(x.timestamp for x in bar_list)
                    first_trade_date = first_ts.date()
                    validated["alpaca_has_daily_bars"] = True
                    validated["alpaca_first_trade_date"] = first_trade_date
                    validated["alpaca_first_trade_days_from_ipo"] = (first_trade_date - ipo_d).days
                    validated["alpaca_bar_count_30d"] = len(bar_list)
                else:
                    validated["alpaca_has_daily_bars"] = False
                    validated["alpaca_first_trade_date"] = None
                    validated["alpaca_first_trade_days_from_ipo"] = None
                    validated["alpaca_bar_count_30d"] = 0
            except Exception as exc:
                validated["alpaca_has_daily_bars"] = False
                validated["alpaca_first_trade_date"] = None
                validated["alpaca_first_trade_days_from_ipo"] = None
                validated["alpaca_bar_count_30d"] = 0
                validated["alpaca_error"] = f"{type(exc).__name__}: {exc}"

        out.append(validated)

        if (idx + 1) % 50 == 0 or idx + 1 == len(rows):
            print(f"[alpaca] validated {idx + 1}/{len(rows)} symbols")

    return out


def _print_summary(df_all: pl.DataFrame, df_keep: pl.DataFrame) -> None:
    print("\nSummary")
    print("-" * 72)
    print(f"Fetched records (pre-parse/filter): {df_all.height}")
    print(f"Final included rows:                {df_keep.height}")
    print(f"Unique symbols:                     {df_keep['symbol'].n_unique() if df_keep.height else 0}")

    if df_keep.height:
        min_d = df_keep["ipo_date"].min()
        max_d = df_keep["ipo_date"].max()
        print(f"IPO date range:                     {min_d} to {max_d}")

    excluded = (
        df_all.filter(pl.col("included") == False)  # noqa: E712
        .group_by("excluded_reasons")
        .len()
        .sort("len", descending=True)
    )
    if excluded.height:
        print("\nTop exclusion buckets:")
        for row in excluded.iter_rows(named=True):
            print(f"  {row['excluded_reasons']}: {row['len']}")


def main() -> None:
    args = _parse_args()
    if not args.api_key:
        raise RuntimeError(
            "Missing Intrinio API key. Set INTRINIO_API_KEY or pass --api-key."
        )

    start = _parse_iso_date(args.start_date)
    end = _parse_iso_date(args.end_date)
    if start is None or end is None:
        raise RuntimeError("--start-date/--end-date must be valid ISO dates (YYYY-MM-DD).")
    if start > end:
        raise RuntimeError("--start-date must be <= --end-date.")

    statuses = [s.strip() for s in args.statuses.split(",") if s.strip()]
    if not statuses:
        raise RuntimeError("--statuses must include at least one value.")

    fetched_at = datetime.now(tz=timezone.utc).isoformat()
    print(
        f"Fetching Intrinio IPOs from {start.isoformat()} to {end.isoformat()} "
        f"for statuses={statuses} ..."
    )

    raw_payloads = fetch_intrinio_ipos(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        statuses=statuses,
        api_key=args.api_key,
        page_size=args.page_size,
        max_pages=args.max_pages,
        verbose=args.verbose,
    )
    print(f"Fetched {len(raw_payloads)} raw IPO records")

    parsed_rows = [parsed for parsed in (parse_ipo_record(x) for x in raw_payloads) if parsed is not None]
    parsed_rows = _dedupe_rows(parsed_rows)
    print(f"Parsed + deduped to {len(parsed_rows)} symbol/date rows")

    kept_rows, all_rows = _apply_filters(
        parsed_rows,
        include_spacs=args.include_spacs,
        allow_non_us_exchanges=args.allow_non_us_exchanges,
        allow_non_common=args.allow_non_common,
        start=start,
        end=end,
    )

    if args.validate_alpaca:
        print("Running optional Alpaca first-trade validation ...")
        kept_rows = _alpaca_validate_first_trade(kept_rows, limit=args.validate_limit)

    for row in kept_rows:
        row["source_provider"] = "intrinio"
        row["source_endpoint"] = INTRINIO_IPOS_URL
        row["source_fetched_at_utc"] = fetched_at
        row["filter_version"] = FILTER_VERSION
        row["build_start_date"] = start
        row["build_end_date"] = end

    for row in all_rows:
        row["source_provider"] = "intrinio"
        row["source_endpoint"] = INTRINIO_IPOS_URL
        row["source_fetched_at_utc"] = fetched_at
        row["filter_version"] = FILTER_VERSION
        row["build_start_date"] = start
        row["build_end_date"] = end

    df_all = pl.DataFrame(all_rows) if all_rows else pl.DataFrame(schema={"symbol": pl.String})
    df_keep = pl.DataFrame(kept_rows) if kept_rows else pl.DataFrame(schema={"symbol": pl.String})
    df_keep = df_keep.sort(["ipo_date", "symbol"]) if df_keep.height else df_keep

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_keep.write_parquet(args.output)
    print(f"Wrote included universe: {args.output} ({df_keep.height} rows)")

    excluded_out = args.output.with_name(f"{args.output.stem}_with_exclusions.parquet")
    df_all.write_parquet(excluded_out)
    print(f"Wrote audit table:       {excluded_out} ({df_all.height} rows)")

    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        df_keep.write_csv(args.csv_output)
        print(f"Wrote CSV:               {args.csv_output}")

    _print_summary(df_all, df_keep)


if __name__ == "__main__":
    main()
