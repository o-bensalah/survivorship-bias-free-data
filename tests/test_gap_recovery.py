"""Tests for the self-healing gap recovery: last_price_date() (ground-truth
read from each ticker's own file) and bucket_tickers_by_gap() (sizing the
re-fetch window to how far behind each ticker actually is)."""
import pandas as pd

import update_data as u
from factories import make_price_df


def test_last_price_date_reads_the_most_recent_row(prices_dir):
    df = make_price_df([
        ("2026-07-08", 10, 11, 9, 10, 10, 100, 0, 0),
        ("2026-07-10", 10, 11, 9, 10, 10, 100, 0, 0),
    ])
    u.save_ticker_frame("XYZ", df)
    assert u.last_price_date("XYZ") == pd.Timestamp("2026-07-10")


def test_last_price_date_is_none_for_a_never_fetched_ticker(prices_dir):
    assert u.last_price_date("NEVERFETCHED") is None


def test_gap_bucketing_by_distance(prices_dir):
    today = pd.Timestamp("2026-07-20")

    u.save_ticker_frame("RECENT", make_price_df([("2026-07-15", 1, 1, 1, 1, 1, 1, 0, 0)]))  # 5 days behind
    u.save_ticker_frame("STALE", make_price_df([("2026-06-20", 1, 1, 1, 1, 1, 1, 0, 0)]))  # 30 days behind
    # "NEVERFETCHED" has no file at all -- never successfully downloaded before.

    buckets = u.bucket_tickers_by_gap({"RECENT", "STALE", "NEVERFETCHED"}, today)

    assert buckets["1mo"] == {"RECENT"}
    assert buckets["6mo"] == {"STALE"}
    assert buckets["max"] == {"NEVERFETCHED"}


def test_gap_bucketing_boundary_is_inclusive(prices_dir):
    """Exactly 10 days behind is the edge of the (10, "1mo") bucket -- must
    land in "1mo", not spill into "6mo"."""
    today = pd.Timestamp("2026-07-20")
    u.save_ticker_frame("EDGE", make_price_df([("2026-07-10", 1, 1, 1, 1, 1, 1, 0, 0)]))

    buckets = u.bucket_tickers_by_gap({"EDGE"}, today)
    assert buckets["1mo"] == {"EDGE"}


def test_very_long_gap_falls_through_to_full_backfill(prices_dir):
    """A ticker that's been failing silently for months (e.g. a bug that went
    unnoticed) must fall into "max" so the whole gap gets recovered, not
    just a bounded recent window."""
    today = pd.Timestamp("2026-07-20")
    u.save_ticker_frame("LONGGAP", make_price_df([("2025-01-01", 1, 1, 1, 1, 1, 1, 0, 0)]))

    buckets = u.bucket_tickers_by_gap({"LONGGAP"}, today)
    assert buckets["max"] == {"LONGGAP"}
