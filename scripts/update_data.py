"""
Daily update of a survivorship-bias-free NYSE + Nasdaq + NYSE American (AMEX)
stock database.

Universe & membership changes: derived by diffing daily snapshots of Nasdaq
Trader's official listed-securities directory (nasdaqlisted.txt +
otherlisted.txt) against the previously tracked universe. Unlike the S&P 500
(whose Wikipedia page has a decades-deep historical change log), there's no
free pre-built delisting/IPO registry for the full market, so bias-free
tracking here starts from whenever this script first runs and grows more
complete every day after that.

Prices, dividends, splits: pulled from Yahoo Finance via yfinance.

Every ticker that has ever been tracked keeps its own price file under
data/prices/, even after it delists, so history is never silently rewritten
to only show today's survivors.
"""
import datetime as dt
import io
import re
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

NASDAQLISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHERLISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
# otherlisted.txt covers multiple trading venues; we only want primary listings
# on NYSE (N) and NYSE American / AMEX (A) -- excludes NYSE Arca (P), Cboe BZX
# (Z), IEXG (V) and Chicago (M), which aren't primary listing exchanges.
KEPT_OTHER_EXCHANGES = {"N", "A"}
EXCHANGE_NAMES = {"N": "NYSE", "A": "NYSE American"}

REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; survivorship-bias-free-data-bot/1.0)"}

# Excludes warrants/rights/units/preferred stock/debt securities, which aren't
# common equity. ETFs are exempted since fund names legitimately contain
# words like "Bond" or "Preferred" (e.g. "iShares Core U.S. Aggregate Bond ETF").
NON_COMMON_PATTERN = re.compile(r"\b(?:Warrants?|Rights?|Units?|Preferred|Notes?|Bonds?|Debentures?)\b", re.IGNORECASE)

UNIVERSE_COLUMNS = ["ticker", "name", "exchange", "financial_status", "status", "first_seen", "last_seen"]
EVENTS_COLUMNS = ["date", "ticker", "event", "detail"]

CHUNK_SIZE = 250
CHUNK_DELAY_SECONDS = 2


def clean_ticker(t: str) -> str:
    # Nasdaq Trader uses "." for share classes (e.g. "BF.B"); Yahoo Finance
    # expects "-" (e.g. "BF-B"). Some tickers (warrants, preferred series) use
    # conventions that don't map cleanly to Yahoo at all -- those simply fail
    # to download and get flagged as NO_DATA rather than special-cased here.
    return str(t).strip().replace(".", "-")


def fetch_listed_file(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    resp.raise_for_status()
    lines = resp.text.splitlines()[:-1]  # drop the "File Creation Time: ..." trailer row
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|")


def fetch_listed_universe() -> pd.DataFrame:
    nasdaq = fetch_listed_file(NASDAQLISTED_URL)
    nasdaq = nasdaq[nasdaq["Test Issue"] != "Y"].copy()
    non_common = nasdaq["Security Name"].str.contains(NON_COMMON_PATTERN, na=False, regex=True) & (nasdaq["ETF"] != "Y")
    nasdaq = nasdaq[~non_common]
    nasdaq = nasdaq.rename(columns={"Symbol": "ticker", "Security Name": "name", "Financial Status": "financial_status"})
    nasdaq["exchange"] = "NASDAQ"
    nasdaq = nasdaq[["ticker", "name", "exchange", "financial_status"]]

    other = fetch_listed_file(OTHERLISTED_URL)
    other = other[(other["Test Issue"] != "Y") & (other["Exchange"].isin(KEPT_OTHER_EXCHANGES))].copy()
    non_common2 = other["Security Name"].str.contains(NON_COMMON_PATTERN, na=False, regex=True) & (other["ETF"] != "Y")
    other = other[~non_common2]
    other = other.rename(columns={"ACT Symbol": "ticker", "Security Name": "name"})
    other["exchange"] = other["Exchange"].map(EXCHANGE_NAMES)
    other["financial_status"] = "N"
    other = other[["ticker", "name", "exchange", "financial_status"]]

    universe = pd.concat([nasdaq, other], ignore_index=True)
    universe = universe.drop_duplicates(subset="ticker", keep="first")
    universe["ticker"] = universe["ticker"].map(clean_ticker)
    return universe.reset_index(drop=True)


def load_universe() -> pd.DataFrame:
    if UNIVERSE_FILE.exists():
        return pd.read_csv(UNIVERSE_FILE, parse_dates=["first_seen", "last_seen"])
    return pd.DataFrame(columns=UNIVERSE_COLUMNS)


def load_events() -> pd.DataFrame:
    if EVENTS_FILE.exists():
        return pd.read_csv(EVENTS_FILE, parse_dates=["date"])
    return pd.DataFrame(columns=EVENTS_COLUMNS)


class EventLog:
    """Buffers new event rows and appends them in one batch at the end,
    instead of growing a DataFrame one row at a time (too slow once the
    universe is thousands of tickers instead of hundreds)."""

    def __init__(self, events: pd.DataFrame):
        self.events = events
        self.seen = set(zip(events["date"], events["ticker"], events["event"]))
        self.new_rows = []

    def log(self, date, ticker, event, detail):
        key = (date, ticker, event)
        if key in self.seen:
            return
        self.seen.add(key)
        self.new_rows.append({"date": date, "ticker": ticker, "event": event, "detail": detail})

    def finalize(self) -> pd.DataFrame:
        if not self.new_rows:
            return self.events
        return pd.concat([self.events, pd.DataFrame(self.new_rows)], ignore_index=True)


def sync_universe(today: pd.Timestamp, listed: pd.DataFrame, universe: pd.DataFrame, log: EventLog):
    current_tickers = set(listed["ticker"])
    known_tickers = set(universe["ticker"]) if not universe.empty else set()
    exchange_map = listed.set_index("ticker")["exchange"]

    new_tickers = current_tickers - known_tickers
    if new_tickers:
        new_rows = listed[listed["ticker"].isin(new_tickers)].copy()
        new_rows["status"] = "active"
        new_rows["first_seen"] = today
        new_rows["last_seen"] = today
        universe = pd.concat([universe, new_rows[UNIVERSE_COLUMNS]], ignore_index=True)
        for t in sorted(new_tickers):
            log.log(today, t, "ADD", f"Newly observed on {exchange_map.get(t, '?')}")

    if not universe.empty:
        dropped_mask = (universe["status"] == "active") & ~universe["ticker"].isin(current_tickers)
        for t in sorted(universe.loc[dropped_mask, "ticker"]):
            log.log(today, t, "REMOVE", "No longer in NYSE/Nasdaq/AMEX listed directory")
        universe.loc[dropped_mask, "status"] = "removed"

        active_mask = universe["ticker"].isin(current_tickers)
        universe.loc[active_mask, "last_seen"] = today
        status_map = listed.set_index("ticker")["financial_status"]
        universe.loc[active_mask, "financial_status"] = universe.loc[active_mask, "ticker"].map(status_map)

    return universe, new_tickers


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


# Reserved Windows device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9) can't be
# used as filenames on Windows even with an extension, e.g. "PRN.csv" -- and
# this repo is meant to be clonable on Windows, not just run on Linux CI.
WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


def price_file_path(ticker: str) -> Path:
    stem = ticker + "_" if ticker.upper() in WINDOWS_RESERVED_NAMES else ticker
    return PRICES / f"{stem}.csv"


def save_ticker_frame(ticker: str, df: pd.DataFrame):
    df = df.dropna(how="all")
    if df.empty:
        return
    df = df.reset_index().rename(columns={"index": "Date"})
    df = round_prices(df)
    PRICES.mkdir(parents=True, exist_ok=True)
    out_path = price_file_path(ticker)
    if out_path.exists():
        existing = pd.read_csv(out_path, parse_dates=["Date"])
        combined = pd.concat([existing, df]).drop_duplicates(subset="Date", keep="last")
    else:
        combined = df
    combined = combined.sort_values("Date")
    combined.to_csv(out_path, index=False)


def chunked(items, size):
    items = sorted(items)
    for i in range(0, len(items), size):
        yield items[i:i + size]


def download_prices(tickers, period):
    if not tickers:
        return set()
    silent_failures = set()
    chunks = list(chunked(tickers, CHUNK_SIZE))
    for i, chunk in enumerate(chunks, 1):
        print(f"  chunk {i}/{len(chunks)} ({len(chunk)} tickers, period={period})")
        data = download_batch(chunk, period)
        if data.empty:
            silent_failures.update(chunk)
            continue
        single = len(chunk) == 1
        for t in chunk:
            try:
                df = data if single else data[t]
            except KeyError:
                silent_failures.add(t)
                continue
            if df.dropna(how="all").empty:
                silent_failures.add(t)
                continue
            save_ticker_frame(t, df)
        if i < len(chunks):
            time.sleep(CHUNK_DELAY_SECONDS)
    return silent_failures


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp(dt.date.today())

    listed = fetch_listed_universe()
    universe = load_universe()
    events = load_events()
    log = EventLog(events)

    universe, new_tickers = sync_universe(today, listed, universe, log)

    active_tickers = set(universe.loc[universe["status"] == "active", "ticker"])
    silent_failures = set()

    if new_tickers:
        print(f"Backfilling full history for {len(new_tickers)} new ticker(s)")
        silent_failures |= download_prices(new_tickers, period="max")

    daily_tickers = active_tickers - new_tickers
    print(f"Fetching latest daily bars for {len(daily_tickers)} ticker(s)")
    silent_failures |= download_prices(daily_tickers, period="5d")

    for t in sorted(silent_failures):
        idx = universe.index[universe["ticker"] == t]
        if len(idx) and universe.loc[idx[0], "status"] == "active":
            log.log(
                today, t, "NO_DATA",
                "Active in universe but Yahoo Finance returned no data (possible delisting)",
            )

    universe.to_csv(UNIVERSE_FILE, index=False)
    events = log.finalize()
    events.sort_values(["date", "ticker"]).to_csv(EVENTS_FILE, index=False)

    print(f"Done. Universe size: {len(universe)}, active: {len(active_tickers)}, events: {len(events)}")


if __name__ == "__main__":
    main()
