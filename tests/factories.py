"""Small builders for the synthetic DataFrames the tests construct by hand,
standing in for a Nasdaq Trader snapshot / universe.csv / events.csv without
touching the network or real files."""
import pandas as pd

import update_data as u


def make_listed(rows):
    """rows: list of (ticker, name, exchange, financial_status) tuples."""
    return pd.DataFrame(rows, columns=["ticker", "name", "exchange", "financial_status"])


def make_universe(rows):
    """rows: list of dicts with keys matching UNIVERSE_COLUMNS."""
    if not rows:
        return pd.DataFrame(columns=u.UNIVERSE_COLUMNS)
    df = pd.DataFrame(rows)
    df["first_seen"] = pd.to_datetime(df["first_seen"])
    df["last_seen"] = pd.to_datetime(df["last_seen"])
    return df[u.UNIVERSE_COLUMNS]


def make_events():
    return pd.DataFrame(columns=u.EVENTS_COLUMNS)


def make_price_df(rows):
    """rows: list of (date, open, high, low, close, adj_close, volume, dividends, splits)
    tuples, shaped like what yfinance returns (DatetimeIndex, not a Date column)."""
    df = pd.DataFrame(
        rows,
        columns=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date")
