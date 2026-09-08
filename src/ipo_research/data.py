"""Fetch and freeze a public-data convenience sample with an exclusion ledger."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import re
from pathlib import Path
import urllib.parse
import urllib.request


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fetch_history(symbol: str, start: date, end: date, cache: Path, refresh: bool = False) -> dict:
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,15}", symbol) or ".." in symbol:
        raise ValueError("Use a valid ticker symbol")
    params = urllib.parse.urlencode({
        "period1": int(datetime.combine(start, datetime.min.time(), timezone.utc).timestamp()),
        "period2": int(datetime.combine(end, datetime.min.time(), timezone.utc).timestamp()),
        "interval": "1d", "events": "div,splits",
    })
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol, safe='')}?{params}"
    path = cache / f"{symbol}-{start}-{end}.json"
    raw_path = path.with_suffix(".response.json")
    if path.exists() and raw_path.exists() and not refresh:
        saved = json.loads(path.read_text())
        if digest(raw_path.read_bytes()) != saved["raw_sha256"]:
            raise ValueError(f"Cached response hash mismatch for {symbol}")
        return saved
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 IPOResearch/2.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read()
    chart = json.loads(raw)["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError(str(chart.get("error") or "No price history"))
    result = chart["result"][0]
    if result["meta"].get("currency") != "USD":
        raise ValueError("Expected USD prices")
    from zoneinfo import ZoneInfo
    zone = ZoneInfo(result["meta"]["exchangeTimezoneName"])
    quotes = result["indicators"]["quote"][0]
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    bars = []
    for i, timestamp in enumerate(result.get("timestamp", [])):
        bars.append({
            "date": datetime.fromtimestamp(timestamp, zone).date().isoformat(),
            "adjusted_close": adjusted[i], "volume": quotes["volume"][i],
        })
    saved = {
        "symbol": symbol, "source_url": url,
        "source_page": f"https://finance.yahoo.com/quote/{symbol}/history/",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": digest(raw),
        "first_trade_date": datetime.fromtimestamp(result["meta"]["firstTradeDate"], zone).date().isoformat(),
        "bars": bars,
    }
    cache.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    path.write_text(json.dumps(saved, indent=2) + "\n")
    return saved


def fetch_cohort(universe_path: Path, output: Path, refresh: bool = False) -> dict:
    universe = json.loads(universe_path.read_text())
    listings = universe["listings"]
    if len({row["symbol"] for row in listings}) != len(listings):
        raise ValueError("The universe must contain one entry per symbol")
    dates = [date.fromisoformat(row["listing_date"]) for row in listings]
    cache = output.parent / "raw"
    benchmark = fetch_history("SPY", min(dates) - timedelta(days=45), max(dates) + timedelta(days=130), cache, refresh)

    def fetch_one(row):
        try:
            listing = date.fromisoformat(row["listing_date"])
            history = fetch_history(row["symbol"], listing - timedelta(days=7), listing + timedelta(days=130), cache, refresh)
            if history["first_trade_date"] != row["listing_date"]:
                raise ValueError(f"Vendor first trade {history['first_trade_date']} differs from registry {row['listing_date']}")
            return {**row, **history}, None
        except Exception as exc:
            return None, {**row, "reason": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=3) as pool:
        fetched = list(pool.map(fetch_one, listings))
    bundle = {
        "schema_version": "2.0.0", "universe_sha256": digest(universe_path.read_bytes()),
        "selection": universe["selection"],
        "limitations": [
            "Curated convenience sample. Missing or delisted histories can create survivorship bias.",
            "Vendor-adjusted prices reflect the current historical revision, not archived point-in-time quotes.",
            "The public chart endpoint has no stability guarantee. Retain this input snapshot for reproduction.",
            "Security type and listing date require an independent source audit before population-level claims.",
        ],
        "benchmark": benchmark,
        "listings": [row for row, error in fetched if row is not None],
        "exclusions": [error for row, error in fetched if error is not None],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    return bundle
