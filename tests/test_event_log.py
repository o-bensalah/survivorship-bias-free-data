"""Tests for EventLog: the batched, deduplicating event writer."""
import pandas as pd

import update_data as u
from factories import make_events


def test_dedups_identical_entries_within_a_run():
    log = u.EventLog(make_events())
    log.log(pd.Timestamp("2026-07-11"), "AAPL", "ADD", "first")
    log.log(pd.Timestamp("2026-07-11"), "AAPL", "ADD", "duplicate, should be dropped")
    result = log.finalize()

    assert len(result) == 1
    assert result.iloc[0]["detail"] == "first"


def test_preloads_seen_keys_from_previously_saved_events():
    """A (date, ticker, event) already present in the loaded events.csv must
    not be re-logged, even by a fresh EventLog built on a later run."""
    existing = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-10"), "ticker": "AAPL", "event": "ADD", "detail": "already there"},
    ])
    log = u.EventLog(existing)
    log.log(pd.Timestamp("2026-07-10"), "AAPL", "ADD", "should be skipped")
    result = log.finalize()

    assert len(result) == 1
    assert result.iloc[0]["detail"] == "already there"


def test_different_event_types_same_ticker_same_day_both_kept():
    log = u.EventLog(make_events())
    log.log(pd.Timestamp("2026-07-11"), "AAPL", "ADD", "a")
    log.log(pd.Timestamp("2026-07-11"), "AAPL", "EXCHANGE_CHANGE", "b")
    result = log.finalize()

    assert len(result) == 2


def test_finalize_with_no_new_rows_returns_original_frame_unchanged():
    existing = pd.DataFrame([
        {"date": pd.Timestamp("2026-07-10"), "ticker": "AAPL", "event": "ADD", "detail": "x"},
    ])
    log = u.EventLog(existing)
    result = log.finalize()
    assert len(result) == 1
