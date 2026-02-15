from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ipo_risk_engine.config.settings import load_settings
from ipo_risk_engine.data.ingest import ingest_daily_bars


def main():
    s = load_settings()
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30)

    res = ingest_daily_bars("PLTR", start=start, end=end, settings=s, force_refresh=False)
    print(res)


if __name__ == "__main__":
    main()
