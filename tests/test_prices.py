"""Tests for save_ticker_frame() and price_file_path(): how dividends,
splits, and reverse splits get persisted, how daily re-fetches merge with
existing history, and the Windows-reserved-filename handling."""
import pandas as pd

import update_data as u
from factories import make_price_df


def test_dividend_is_preserved(prices_dir):
    df = make_price_df([
        ("2026-07-10", 100.0, 101.0, 99.0, 100.5, 99.7, 1_000_000, 0.24, 0.0),
    ])
    u.save_ticker_frame("KO", df)

    saved = pd.read_csv(u.price_file_path("KO"))
    assert saved.loc[0, "Dividends"] == 0.24


def test_forward_split_is_preserved(prices_dir):
    """e.g. a 4-for-1 split: the Stock Splits column carries the ratio (4.0)
    on the split date, same as yfinance reports it."""
    df = make_price_df([
        ("2020-08-31", 500.0, 510.0, 490.0, 500.0, 500.0, 5_000_000, 0.0, 4.0),
    ])
    u.save_ticker_frame("AAPL", df)

    saved = pd.read_csv(u.price_file_path("AAPL"))
    assert saved.loc[0, "Stock Splits"] == 4.0


def test_reverse_split_is_preserved(prices_dir):
    """e.g. a 1-for-10 reverse split: ratio is < 1 (0.1). Same column, same
    code path as a forward split -- this just pins the expected value."""
    df = make_price_df([
        ("2026-07-10", 0.17, 0.18, 0.16, 0.17, 0.17, 100_000, 0.0, 0.1),
    ])
    u.save_ticker_frame("GIPR", df)

    saved = pd.read_csv(u.price_file_path("GIPR"))
    assert saved.loc[0, "Stock Splits"] == 0.1


def test_merge_keeps_history_and_newer_fetch_wins_on_overlap(prices_dir):
    """Simulates two daily runs whose fetch windows overlap by one day: the
    older row's data on the overlapping date is replaced by the newer
    fetch, and the previously saved day is never lost."""
    day1 = make_price_df([("2026-07-08", 10, 11, 9, 10.5, 10.5, 100, 0.0, 0.0)])
    u.save_ticker_frame("XYZ", day1)

    day2 = make_price_df([
        ("2026-07-08", 10, 11, 9, 10.6, 10.6, 150, 0.0, 0.0),  # revised
        ("2026-07-09", 10.6, 12, 10, 11.5, 11.5, 200, 0.0, 0.0),  # new day
    ])
    u.save_ticker_frame("XYZ", day2)

    saved = pd.read_csv(u.price_file_path("XYZ"), parse_dates=["Date"])
    assert len(saved) == 2
    assert saved.loc[saved["Date"] == "2026-07-08", "Close"].iloc[0] == 10.6
    assert saved.loc[saved["Date"] == "2026-07-09", "Close"].iloc[0] == 11.5


def test_full_precision_prices_are_rounded(prices_dir):
    df = make_price_df([
        ("2026-07-10", 0.1283479928970337, 0.1289059966802597,
         0.1283479928970337, 0.1283479928970337, 0.0982070341706276,
         469033600, 0.0, 0.0),
    ])
    u.save_ticker_frame("AAPL", df)

    saved = pd.read_csv(u.price_file_path("AAPL"))
    assert saved.loc[0, "Open"] == 0.128348


def test_empty_fetch_does_not_create_a_file(prices_dir):
    df = make_price_df([])
    u.save_ticker_frame("NODATA", df)
    assert not u.price_file_path("NODATA").exists()


def test_windows_reserved_ticker_names_get_safe_filenames(prices_dir):
    for ticker in ("CON", "PRN", "con", "Prn", "NUL", "COM1", "LPT9"):
        path = u.price_file_path(ticker)
        assert path.stem.upper().rstrip("_") == ticker.upper()
        assert path.stem.endswith("_")


def test_ordinary_ticker_names_are_untouched(prices_dir):
    assert u.price_file_path("AAPL").name == "AAPL.csv"
    assert u.price_file_path("NA").name == "NA.csv"  # regression: was corrupted to "nan"
