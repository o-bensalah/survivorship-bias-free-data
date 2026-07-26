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


def _looks_like_split_drift(stored_row, live_row) -> bool:
    """Yahoo retroactively rescales its *raw* OHLC/Volume history (not just
    Adj Close) once a split's adjustment propagates through its backend --
    confirmed via SNAL's 2026-07-06 reverse split, where a fresh fetch of a
    2023 date came back exactly 5x our previously-stored value. Our pipeline
    only re-fetches a rolling window, so an older row stays frozen at the
    pre-split scale until something re-touches that date. That's expected
    staleness, not corruption -- detected here via a single consistent
    scaling factor across Open/High/Low/Close with Volume moving by the
    inverse factor, which is what a clean split (and only a split) produces.
    Random data corruption wouldn't line up this precisely across 5 columns.
    """
    stored_close, live_close = float(stored_row["Close"]), float(live_row["Close"])
    if stored_close == 0 or live_close == 0:
        return False
    ratio = live_close / stored_close
    if abs(ratio - 1) < 0.02:
        return False  # not drifted at all

    for col in ("Open", "High", "Low"):
        stored_val, live_val = float(stored_row[col]), float(live_row[col])
        if stored_val == 0 or abs(live_val / stored_val - ratio) > 0.05 * abs(ratio):
            return False

    stored_vol, live_vol = float(stored_row["Volume"]), float(live_row["Volume"])
    if stored_vol == 0 or live_vol == 0:
        return False
    volume_ratio = stored_vol / live_vol
    if abs(volume_ratio - ratio) > 0.05 * abs(ratio):
        return False

    return True


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
    split_drifted = []
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

        if _looks_like_split_drift(stored_row, live_row):
            split_drifted.append(f"{ticker} {date.date()}")
        else:
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
    if split_drifted:
        print(
            f"OHLC/Volume drifted from live for {len(split_drifted)}/{len(picks)} sampled rows, "
            f"consistent with a split that occurred after the row was saved "
            f"(Yahoo retroactively rescales raw prices, not just Adj Close): {split_drifted}"
        )
    if adj_close_drifted:
        print(
            f"Adj Close drifted from live for {len(adj_close_drifted)}/{len(picks)} sampled rows "
            f"(expected if a split/dividend occurred since these rows were saved): {adj_close_drifted}"
        )

    assert not mismatches, "Spot-check found mismatches vs live Yahoo data:\n" + "\n".join(mismatches)
