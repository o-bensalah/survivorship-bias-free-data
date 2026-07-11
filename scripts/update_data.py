"""
Daily update of a survivorship-bias-free S&P 500 database.

- Universe & membership changes: scraped from Wikipedia's
  "List of S&P 500 companies" page (current constituents + dated
  history of index additions/removals).
- Prices, dividends, splits: pulled from Yahoo Finance via yfinance.

Every ticker that has ever been tracked keeps its own price file under
data/prices/, even after it leaves the index or gets delisted, so the
history is never silently rewritten to only show today's survivors.
"""
import datetime as dt
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRICES = DATA / "prices"
UNIVERSE_FILE = DATA / "universe.csv"
EVENTS_FILE = DATA / "events.csv"

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia rejects requests without a browser-like User-Agent (403).
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; survivorship-bias-free-data-bot/1.0)"}

UNIVERSE_COLUMNS = ["ticker", "name", "sector", "sub_industry", "status", "first_seen", "last_seen"]
EVENTS_COLUMNS = ["date", "ticker", "event", "detail"]


def clean_ticker(t: str) -> str:
    # Wikipedia uses "BRK.B" style, Yahoo Finance expects "BRK-B".
    return str(t).strip().replace(".", "-")


def fetch_wikipedia_tables():
    resp = requests.get(WIKI_URL, headers=REQUEST_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    constituents = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    constituents.columns = ["ticker", "name", "sector", "sub_industry"]
    constituents["ticker"] = constituents["ticker"].map(clean_ticker)

    changes = tables[1].copy()
    changes.columns = ["date", "added_ticker", "added_name", "removed_ticker", "removed_name", "reason"]
    # First data row can be a repeated header (e.g. "Effective Date") depending on the page's markup.
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    changes = changes.dropna(subset=["date"])
    for col in ("added_ticker", "removed_ticker"):
        changes[col] = changes[col].map(lambda x: clean_ticker(x) if pd.notna(x) else x)
    return constituents, changes


def load_universe() -> pd.DataFrame:
    if UNIVERSE_FILE.exists():
        return pd.read_csv(UNIVERSE_FILE, parse_dates=["first_seen", "last_seen"])
    return pd.DataFrame(columns=UNIVERSE_COLUMNS)


def load_events() -> pd.DataFrame:
    if EVENTS_FILE.exists():
        return pd.read_csv(EVENTS_FILE, parse_dates=["date"])
    return pd.DataFrame(columns=EVENTS_COLUMNS)


def log_event(events: pd.DataFrame, date, ticker, event, detail) -> pd.DataFrame:
    already_logged = (
        (events["date"] == date) & (events["ticker"] == ticker) & (events["event"] == event)
    ).any()
    if already_logged:
        return events
    events.loc[len(events)] = [date, ticker, event, detail]
    return events


def sync_index_changes(changes: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Backfill every dated ADD/REMOVE row from Wikipedia's change log that we
    haven't recorded yet. Safe to run every day: new rows Wikipedia adds
    over time get picked up, already-logged rows are skipped."""
    for _, row in changes.iterrows():
        if pd.notna(row["added_ticker"]):
            events = log_event(events, row["date"], row["added_ticker"], "ADD", "S&P 500 addition")
        if pd.notna(row["removed_ticker"]):
            events = log_event(events, row["date"], row["removed_ticker"], "REMOVE", "S&P 500 removal")
    return events


def update_universe_and_events(today: pd.Timestamp):
    constituents, changes = fetch_wikipedia_tables()
    universe = load_universe()
    events = load_events()

    events = sync_index_changes(changes, events)

    current_tickers = set(constituents["ticker"])
    known_tickers = set(universe["ticker"]) if not universe.empty else set()

    new_tickers = current_tickers - known_tickers
    for t in sorted(new_tickers):
        row = constituents.loc[constituents["ticker"] == t].iloc[0]
        universe.loc[len(universe)] = [
            t, row["name"], row["sector"], row["sub_industry"], "active", today, today,
        ]
        events = log_event(events, today, t, "ADD", "Newly observed in tracked universe")

    if not universe.empty:
        dropped = known_tickers - current_tickers
        for t in sorted(dropped):
            idx = universe.index[universe["ticker"] == t]
            if len(idx) and universe.loc[idx[0], "status"] == "active":
                universe.loc[idx[0], "status"] = "removed"
                events = log_event(events, today, t, "REMOVE", "No longer in S&P 500 constituent list")

        active_mask = universe["ticker"].isin(current_tickers)
        universe.loc[active_mask, "last_seen"] = today

    universe.to_csv(UNIVERSE_FILE, index=False)
    events.sort_values(["date", "ticker"]).to_csv(EVENTS_FILE, index=False)
    return universe, events, new_tickers


def download_batch(tickers, period, retries=3):
    tickers = sorted(tickers)
    if not tickers:
        return pd.DataFrame()
    for attempt in range(1, retries + 1):
        try:
            return yf.download(
                tickers=tickers,
                period=period,
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                actions=True,
                threads=True,
                progress=False,
            )
        except Exception as exc:  # network / rate-limit hiccups
            if attempt == retries:
                raise
            wait = 10 * attempt
            print(f"download failed ({exc}), retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    return pd.DataFrame()


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]


def round_prices(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance returns full float64 precision (16+ significant digits) which
    # bloats the CSVs for no benefit; 6 decimals is far finer than real quotes.
    for col in PRICE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].round(6)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].astype("Int64")
    if "Dividends" in df.columns:
        df["Dividends"] = df["Dividends"].round(6)
    if "Stock Splits" in df.columns:
        df["Stock Splits"] = df["Stock Splits"].round(6)
    return df


def save_ticker_frame(ticker: str, df: pd.DataFrame):
    df = df.dropna(how="all")
    if df.empty:
        return
    df = df.reset_index().rename(columns={"index": "Date"})
    df = round_prices(df)
    PRICES.mkdir(parents=True, exist_ok=True)
    out_path = PRICES / f"{ticker}.csv"
    if out_path.exists():
        existing = pd.read_csv(out_path, parse_dates=["Date"])
        combined = pd.concat([existing, df]).drop_duplicates(subset="Date", keep="last")
    else:
        combined = df
    combined = combined.sort_values("Date")
    combined.to_csv(out_path, index=False)


def download_prices(tickers, period):
    if not tickers:
        return set()
    data = download_batch(tickers, period)
    if data.empty:
        return set()

    single = len(tickers) == 1
    silent_failures = set()
    for t in sorted(tickers):
        try:
            df = data if single else data[t]
        except KeyError:
            silent_failures.add(t)
            continue
        if df.dropna(how="all").empty:
            silent_failures.add(t)
            continue
        save_ticker_frame(t, df)
    return silent_failures


def flag_possible_delistings(universe, events, today, silent_failures):
    if not silent_failures:
        return events
    for t in sorted(silent_failures):
        idx = universe.index[universe["ticker"] == t]
        if len(idx) and universe.loc[idx[0], "status"] == "active":
            events = log_event(
                events, today, t, "NO_DATA",
                "Active in universe but Yahoo Finance returned no data (possible delisting)",
            )
    return events


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    today = pd.Timestamp(dt.date.today())
    universe, events, new_tickers = update_universe_and_events(today)

    active_tickers = set(universe.loc[universe["status"] == "active", "ticker"])
    silent_failures = set()

    if new_tickers:
        print(f"Backfilling full history for {len(new_tickers)} new ticker(s): {sorted(new_tickers)}")
        silent_failures |= download_prices(new_tickers, period="max")

    daily_tickers = active_tickers - new_tickers
    print(f"Fetching latest daily bars for {len(daily_tickers)} ticker(s)")
    silent_failures |= download_prices(daily_tickers, period="5d")

    events = flag_possible_delistings(universe, events, today, silent_failures)
    events.sort_values(["date", "ticker"]).to_csv(EVENTS_FILE, index=False)

    print(f"Done. Universe size: {len(universe)}, active: {len(active_tickers)}, events: {len(events)}")


if __name__ == "__main__":
    main()
