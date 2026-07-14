"""Live spot-check: randomly samples (ticker, date) pairs from the actually
committed data/prices/ files and compares them against a fresh live fetch
from Yahoo Finance. This is the one test that touches real, currently
committed data instead of synthetic fixtures -- it exists to catch data
corruption the offline unit tests can't see (wrong ticker mapped, a bad
merge, a stale/incorrect value), not to test the pipeline's logic.

Excluded from the default `pytest` run (see pytest.ini's `-m "not live"`)
since it needs network access, is slower, and repeated live Yahoo calls on
every push would contribute to the exact rate-limiting risk documented
elsewhere in this project. Run explicitly with `pytest -m live`, or via the
separate weekly spotcheck.yml workflow.
"""
import random
import sys
from pathlib import Path

import pandas as pd
import pytest
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import update_data as u  # noqa: E402

pytestmark = pytest.mark.live

SAMPLE_SIZE = 15
# Exchanges occasionally revise preliminary volume/close figures for a day
# or two after the fact; excluding very recent dates avoids flagging that
# as a false mismatch.
MIN_AGE_DAYS = 7
TOLERANCE = 1e-4
COMPARE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits"]


def _ticker_from_filename(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_") and stem[:-1].upper() in u.WINDOWS_RESERVED_NAMES:
        return stem[:-1]
    return stem


def _sample_ticker_dates(n, min_age_days):
    files = sorted(u.PRICES.glob("*.csv"))
    if not files:
        pytest.skip("no committed price data to spot-check (data/prices/ is empty)")
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=min_age_days)
    picks = []
    attempts = 0
    while len(picks) < n and attempts < n * 20:
        attempts += 1
        path = random.choice(files)
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
        except Exception:
            continue
        eligible = df[df["Date"] <= cutoff]
        if eligible.empty:
            continue
        row = eligible.sample(1).iloc[0]
        picks.append((_ticker_from_filename(path), row))
    return picks


def test_random_spot_check_against_live_yahoo_data():
    picks = _sample_ticker_dates(SAMPLE_SIZE, MIN_AGE_DAYS)
    mismatches = []
    adj_close_drifted = []

    for ticker, stored_row in picks:
        date = stored_row["Date"]
        live = yf.Ticker(ticker).history(
            start=date, end=date + pd.Timedelta(days=1),
            auto_adjust=False, actions=True,
        )
        if live.empty:
            mismatches.append(f"{ticker} {date.date()}: live fetch returned no data")
            continue
        live_row = live.iloc[0]

        for col in COMPARE_COLUMNS:
            stored_val, live_val = stored_row[col], live_row[col]
            if abs(float(stored_val) - float(live_val)) > TOLERANCE:
                mismatches.append(f"{ticker} {date.date()} {col}: stored={stored_val} live={live_val}")

        # Adj Close is expected to drift once a split/dividend happens after
        # a row was originally saved -- a documented limitation (see
        # events.csv discussion), not a bug. Reported for visibility, not
        # asserted on, so it doesn't fail the test.
        if abs(float(stored_row["Adj Close"]) - float(live_row["Adj Close"])) > TOLERANCE:
            adj_close_drifted.append(f"{ticker} {date.date()}")

    print(f"\nSpot-checked {len(picks)} (ticker, date) pairs against live Yahoo data.")
    if adj_close_drifted:
        print(
            f"Adj Close drifted from live for {len(adj_close_drifted)}/{len(picks)} sampled rows "
            f"(expected if a split/dividend occurred since these rows were saved): {adj_close_drifted}"
        )

    assert not mismatches, "Spot-check found mismatches vs live Yahoo data:\n" + "\n".join(mismatches)
