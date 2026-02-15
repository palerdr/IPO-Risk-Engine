from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.timeframe import TimeFrameUnit
from alpaca.data import DataFeed
from ipo_risk_engine.config.settings import AlpacaSettings


@dataclass(frozen=True)
class AlpacaData:
    """
    Thin wrapper around Alpaca's market data client to centralize auth + feed selection
    Keep Alpaca-specific objects isolated from the rest of codebase
    """
    client: StockHistoricalDataClient
    feed: str

    @classmethod
    def from_settings(cls, s: AlpacaSettings) -> "AlpacaData":
        c = StockHistoricalDataClient(api_key=s.api_key, secret_key=s.api_secret)
        return cls(client=c, feed=s.data_feed)
    
    def get_bars(self, symbol: str, tf: TimeFrame, start, end):
        request = StockBarsRequest(
            symbol_or_symbols= symbol,
            timeframe= tf,
            start = start,
            end = end,
            feed = DataFeed.SIP
        )
        return self.client.get_stock_bars(request)

    # def get_daily_bars(self, symbol: str, start, end):
    #     """
    #     Fetch daily bars for a symbol between [start, end).

    #     start/end should be datetime-like (we'll provide them from a script).
    #     """
    #     req = StockBarsRequest(
    #         symbol_or_symbols=symbol,
    #         timeframe=TimeFrame.Day,
    #         start=start,
    #         end=end,
    #         feed=self.feed,
    #     )