from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass(frozen=True)
class AlpacaSettings:
    api_key: str
    api_secret: str
    base_url: str
    data_feed: str


def load_settings() -> AlpacaSettings:
    """
    Loads settings from environment variables.
    """
    # Loads .env into os.environ when running locally.
    # In production, you typically won't use .env; you'll set real environment variables instead.
    load_dotenv(override=False)

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    data_feed = os.getenv("ALPACA_DATA_FEED", "sip")

    missing = []
    if not api_key:
        missing.append("ALPACA_API_KEY")
    if not api_secret:
        missing.append("ALPACA_API_SECRET")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Add them to your .env (local) or set them in your environment."
        )

    return AlpacaSettings(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        data_feed=data_feed,
    )
