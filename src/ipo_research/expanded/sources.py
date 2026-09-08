"""Freeze public sources and preserve rejected records for coverage audits."""

import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import xlrd
from bs4 import BeautifulSoup

from ..data import fetch_history
from .storage import load_json, save_json, sha

RITTER_URL = "https://site.warrington.ufl.edu/ritter/files/IPO-age.xlsx"
SCOOP_URL = "https://www.iposcoop.com/wp-content/uploads/2013/11/SCOOP-Rating-Performance.xls"
DIRECT_SOURCE = "https://site.warrington.ufl.edu/ritter/files/Direct-Listings.pdf"
# Ritter Table 13a identifies these direct listings through December 2025.
DIRECT_LISTINGS = set(
    "SPOT WTRE WORK ASAN PLTR THRY RBLX COIN SQSP ZIP AMPL WRBY BGXX SRFM PODC AIRE FBLG ZENA DMN CSAI NTHI ARAI TTRX SEV OWLS NOMA MEHA WSHP".split()
)


def download(url: str, path: Path) -> dict:
    manifest = path.with_suffix(path.suffix + ".source.json")
    if path.exists() and manifest.exists():
        record = json.loads(manifest.read_text())
        if record["sha256"] != sha(path) or record["url"] != url:
            raise ValueError(f"Source cache mismatch: {path}")
        return record
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 IPOResearch/3.0"})
    with urllib.request.urlopen(request, timeout=35) as response:
        content = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    record = {
        "url": url,
        "sha256": sha(path),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "file": path.name,
    }
    save_json(manifest, record)
    return record


def xlsx_rows(path: Path) -> list[dict]:
    """Read the source's first worksheet without executing workbook formulas."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as book:
        strings = [
            "".join(node.itertext()) for node in ET.fromstring(book.read("xl/sharedStrings.xml"))
        ]
        result = []
        for row in ET.fromstring(book.read("xl/worksheets/sheet1.xml")).findall(".//m:row", ns):
            values = {}
            for cell in row:
                value = cell.find("m:v", ns)
                column = re.sub(r"\d", "", cell.attrib["r"])
                values[column] = (
                    (strings[int(value.text)] if cell.attrib.get("t") == "s" else value.text)
                    if value is not None
                    else ""
                )
            result.append(values)
        return result


def number(value):
    try:
        result = float(str(value).replace("$", "").replace(",", "").strip())
        return result if result > -99 else None
    except (ValueError, TypeError):
        return None


def parse_ritter(path: Path, first_year=2010, last_year=2025) -> list[dict]:
    result = []
    for index, row in enumerate(xlsx_rows(path)[1:], 2):
        raw_date = row.get("A", "")
        if not re.fullmatch(r"\d{8}", raw_date):
            continue
        day = datetime.strptime(raw_date, "%Y%m%d").date()
        if not first_year <= day.year <= last_year:
            continue
        symbol = row.get("C", "").strip().upper()
        result.append(
            {
                "id": f"ritter-{index}",
                "symbol": symbol,
                "name": row.get("B", ""),
                "offer_date": day.isoformat(),
                "cusip": row.get("D") or None,
                "permno": row.get("J") if row.get("J", "").isdigit() else None,
                "adr_code": number(row.get("E")),
                "vc": number(row.get("F")),
                "dual_class": number(row.get("G")),
                "founding_year": number(row.get("K")),
                "source_row": index,
                "source_url": RITTER_URL,
            }
        )
    if len({row["id"] for row in result}) != len(result):
        raise ValueError("Duplicate registry row IDs")
    return result


def parse_scoop(path: Path) -> list[dict]:
    book = xlrd.open_workbook(str(path))
    sheet = book.sheet_by_index(0)
    if sheet.cell_value(35, 2) != "Symbol":
        raise ValueError("IPOScoop workbook schema changed")
    rows = []
    for index in range(36, sheet.nrows):
        values = sheet.row_values(index)
        if not isinstance(values[0], (float, int)) or values[0] < 36000 or not values[2]:
            continue
        day = xlrd.xldate_as_datetime(values[0], book.datemode).date().isoformat()
        rows.append(
            {
                "symbol": str(values[2]).strip().upper(),
                "listing_date": day,
                "offer_price": number(values[4]),
                "underwriters": str(values[3]).strip(),
                "source_url": SCOOP_URL,
                "source_row": index + 1,
            }
        )
    return rows


def parse_stockanalysis(content: str, source_url: str) -> list[dict]:
    soup = BeautifulSoup(content, "html.parser")
    table = soup.select_one("table")
    if table is None:
        raise ValueError(f"No IPO table: {source_url}")
    headers = [cell.get_text(" ", strip=True) for cell in table.select("thead th")]
    if headers[:4] != ["IPO Date", "Symbol", "Company Name", "IPO Price"]:
        raise ValueError(f"IPO table schema changed: {headers}")
    result = []
    for index, row in enumerate(table.select("tbody tr"), 1):
        cells = row.select("td")
        if len(cells) < 4:
            continue
        values = [cell.get_text(" ", strip=True) for cell in cells]
        try:
            day = datetime.strptime(values[0], "%b %d, %Y").date().isoformat()
        except ValueError:
            continue
        result.append(
            {
                "symbol": values[1],
                "name": values[2],
                "listing_date": day,
                "offer_price": number(values[3]),
                "source_url": source_url,
                "source_row": index,
            }
        )
    return result


def registry(directory: Path, first_year=2010, last_year=2025) -> dict:
    raw = directory / "raw"
    sources = [
        download(RITTER_URL, raw / "IPO-age.xlsx"),
        download(SCOOP_URL, raw / "SCOOP-Rating-Performance.xls"),
        download(DIRECT_SOURCE, raw / "Direct-Listings.pdf"),
    ]
    listings = parse_ritter(raw / "IPO-age.xlsx", first_year, last_year)
    supplemental = parse_scoop(raw / "SCOOP-Rating-Performance.xls")
    for year in range(max(2019, first_year), last_year + 1):
        page = 1
        year_rows = []
        while True:
            url = f"https://stockanalysis.com/ipos/{year}/" + (f"?page={page}" if page > 1 else "")
            path = raw / f"stockanalysis-{year}-{page}.html"
            sources.append(download(url, path))
            rows = parse_stockanalysis(path.read_text(), url)
            if any(not row["listing_date"].startswith(str(year)) for row in rows):
                raise ValueError(f"Unexpected IPO year at {url}")
            if page > 1 and {(r["symbol"], r["listing_date"]) for r in rows} & {
                (r["symbol"], r["listing_date"]) for r in year_rows
            }:
                raise ValueError(f"Pagination repeated records at {url}")
            year_rows.extend(rows)
            if len(rows) < 500:
                break
            page += 1
        print(f"Registry {year}: {len(year_rows)} Stock Analysis rows", flush=True)
        supplemental.extend(year_rows)
    lookup = defaultdict(list)
    for row in supplemental:
        lookup[row["symbol"]].append(row)
    for row in listings:
        offer = date.fromisoformat(row["offer_date"])
        matches = [
            r
            for r in lookup[row["symbol"]]
            if abs((date.fromisoformat(r["listing_date"]) - offer).days) <= 4
        ]
        row["cross_references"] = matches
        prices = [
            r["offer_price"]
            for r in matches
            if r["offer_price"] is not None and r["offer_price"] > 0
        ]
        row["offer_price"] = prices[0] if prices and max(prices) - min(prices) <= 0.02 else None
        row["offer_price_conflict"] = bool(prices and max(prices) - min(prices) > 0.02)
        row["underwriters"] = next(
            (r["underwriters"] for r in matches if r.get("underwriters")), None
        )
        row["scope_exclusion"] = scope_exclusion(row)
    bundle = {
        "schema_version": "3.0.0",
        "years": [first_year, last_year],
        "sources": sources,
        "registry": listings,
        "supplemental_count": len(supplemental),
    }
    save_json(directory / "universe.json.gz", bundle)
    return bundle


def scope_exclusion(row: dict) -> str | None:
    symbol, name = row["symbol"], row["name"]
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,15}", symbol) or ".." in symbol:
        return "invalid_or_missing_ticker"
    if row["adr_code"] not in (1, 2):
        return "unsupported_security_code"
    if symbol in DIRECT_LISTINGS:
        return "direct_listing"
    if re.search(r"\b(acquisition|acquisitions|acq|blank.check|spac)\b", name, flags=re.I):
        return "acquisition_company_name"
    if (len(symbol) >= 5 and symbol.endswith("U")) or symbol.endswith((".U", ".W", ".R")):
        return "unit_warrant_or_right_ticker"
    return None


def fetch_prices(directory: Path, workers=4, retry_failures=False) -> dict:
    universe = load_json(directory / "universe.json.gz")
    rows = universe["registry"]
    raw = directory / "raw"
    benchmark = fetch_history(
        "SPY", date(universe["years"][0] - 1, 1, 1), date(universe["years"][1] + 1, 5, 1), raw
    )
    eligible = [r for r in rows if not r["scope_exclusion"]]
    progress = directory / "download-ledger.json"
    ledger = json.loads(progress.read_text()) if progress.exists() else {}

    def one(row):
        cached = ledger.get(row["id"])
        if cached and (cached["status"] == "accepted" or not retry_failures):
            return row["id"], cached
        day = date.fromisoformat(row["offer_date"])
        result = None
        for attempt in range(3):
            try:
                result = fetch_history(
                    row["symbol"].replace(".", "-"),
                    day - timedelta(days=7),
                    day + timedelta(days=160),
                    raw,
                )
                first = date.fromisoformat(result["first_trade_date"])
                if not 0 <= (first - day).days <= 4:
                    raise ValueError(f"ticker_identity_date_mismatch: {first} vs {day}")
                references = {r["listing_date"] for r in row["cross_references"]}
                if references and first.isoformat() not in references:
                    raise ValueError(f"cross_source_date_mismatch: {first} vs {sorted(references)}")
                return row["id"], {"status": "accepted", "history": result}
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                if getattr(exc, "code", None) in (404, 400, 422):
                    break
                if attempt < 2:
                    time.sleep(1 + attempt)
            except (ValueError, KeyError) as exc:
                return row["id"], {"status": "excluded", "reason": str(exc)}
        return row["id"], {
            "status": "excluded",
            "reason": "price_download_failed",
            "history": result,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, row) for row in eligible]
        for index, future in enumerate(as_completed(futures), 1):
            key, value = future.result()
            ledger[key] = value
            if index % 50 == 0 or index == len(futures):
                save_json(progress, ledger)
                counts = Counter(r["status"] for r in ledger.values())
                print(f"Prices {index}/{len(eligible)}: {dict(counts)}", flush=True)
    accepted, excluded = [], []
    for row in rows:
        result = ledger.get(row["id"])
        if row["scope_exclusion"]:
            excluded.append({**row, "reason": row["scope_exclusion"], "exclusion_stage": "scope"})
        elif result["status"] == "accepted":
            accepted.append(
                {
                    **row,
                    "history": result["history"],
                    "listing_date": result["history"]["first_trade_date"],
                }
            )
        else:
            excluded.append({**row, "reason": result["reason"], "exclusion_stage": "prices"})
    bundle = {
        "schema_version": "3.0.0",
        "universe_sha256": sha(directory / "universe.json.gz"),
        "sources": universe["sources"],
        "benchmark": benchmark,
        "listings": accepted,
        "exclusions": excluded,
    }
    save_json(directory / "input.json.gz", bundle)
    return bundle
