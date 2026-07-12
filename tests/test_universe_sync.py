"""Tests for sync_universe(): the day-over-day diff between a Nasdaq Trader
snapshot and the previously tracked universe, which is where IPOs,
delistings, relistings and exchange moves all get detected and logged.

These use synthetic snapshots rather than live data on purpose: the real
Nasdaq Trader files and Yahoo Finance change every day, which would make
tests slow, flaky, and non-reproducible. A few scenarios are inspired by
real events (Twitter's 2022 take-private delisting) but built as
deterministic fixtures, not live lookups -- and SpaceX, unlike the other
examples, hasn't actually IPO'd, so a synthetic ticker stands in for it.
"""
import pandas as pd

import update_data as u
from factories import make_events, make_listed, make_universe

TODAY = pd.Timestamp("2026-07-15")
YESTERDAY = pd.Timestamp("2026-07-14")


def test_ipo_adds_new_ticker_as_active():
    """A brand-new company going public (e.g. a hypothetical SpaceX IPO)
    should appear as a new active row with an ADD event dated today."""
    listed = make_listed([("SPCX", "Rocket Co Common Stock", "NASDAQ", "N")])
    universe = make_universe([])
    log = u.EventLog(make_events())

    universe, new_tickers = u.sync_universe(TODAY, listed, universe, log)
    events = log.finalize()

    assert new_tickers == {"SPCX"}
    row = universe[universe["ticker"] == "SPCX"].iloc[0]
    assert row["status"] == "active"
    assert row["first_seen"] == TODAY
    assert row["last_seen"] == TODAY

    add_events = events[(events["ticker"] == "SPCX") & (events["event"] == "ADD")]
    assert len(add_events) == 1
    assert add_events.iloc[0]["date"] == TODAY


def test_ipo_does_not_also_log_a_spurious_exchange_change():
    """A brand-new ticker's exchange comes straight from the listed snapshot
    on creation -- it shouldn't also fire an EXCHANGE_CHANGE event just
    because it's technically new to the exchange map."""
    listed = make_listed([("SPCX", "Rocket Co Common Stock", "NASDAQ", "N")])
    log = u.EventLog(make_events())

    u.sync_universe(TODAY, listed, make_universe([]), log)
    events = log.finalize()

    assert events[events["event"] == "EXCHANGE_CHANGE"].empty


def test_delisting_marks_ticker_removed_but_keeps_the_row():
    """A company going private (e.g. Twitter's 2022 take-private) drops out
    of the listed-securities directory. It should flip to "removed" and log
    a REMOVE event -- but the row (and its price history) must stay in the
    database, never be deleted."""
    universe = make_universe([
        dict(ticker="TWTR", name="Twitter Inc Common Stock", exchange="NYSE",
             financial_status="N", status="active", first_seen=YESTERDAY, last_seen=YESTERDAY),
    ])
    listed = make_listed([])  # TWTR no longer in today's directory
    log = u.EventLog(make_events())

    universe, new_tickers = u.sync_universe(TODAY, listed, universe, log)
    events = log.finalize()

    assert new_tickers == set()
    assert len(universe) == 1  # row preserved, not deleted
    row = universe[universe["ticker"] == "TWTR"].iloc[0]
    assert row["status"] == "removed"
    # last_seen stays the last day it was actually confirmed listed.
    assert row["last_seen"] == YESTERDAY

    remove_events = events[(events["ticker"] == "TWTR") & (events["event"] == "REMOVE")]
    assert len(remove_events) == 1
    assert remove_events.iloc[0]["date"] == TODAY

    active_tickers = set(universe.loc[universe["status"] == "active", "ticker"])
    assert "TWTR" not in active_tickers


def test_relisting_reactivates_a_previously_removed_ticker():
    """A ticker marked removed that reappears (e.g. emerging from bankruptcy
    under the same symbol) must resume being tracked, not stay permanently
    abandoned just because it was once marked removed."""
    universe = make_universe([
        dict(ticker="OLDCO", name="Old Co Common Stock", exchange="NYSE",
             financial_status="N", status="removed",
             first_seen=pd.Timestamp("2020-01-01"), last_seen=pd.Timestamp("2020-06-01")),
    ])
    listed = make_listed([("OLDCO", "Old Co Common Stock", "NYSE", "N")])
    log = u.EventLog(make_events())

    universe, new_tickers = u.sync_universe(TODAY, listed, universe, log)
    events = log.finalize()

    assert new_tickers == set()  # not "new" -- ticker was already known
    row = universe[universe["ticker"] == "OLDCO"].iloc[0]
    assert row["status"] == "active"
    assert row["last_seen"] == TODAY
    assert row["first_seen"] == pd.Timestamp("2020-01-01")  # provenance kept

    readded_events = events[(events["ticker"] == "OLDCO") & (events["event"] == "READDED")]
    assert len(readded_events) == 1


def test_exchange_switch_updates_field_and_logs_event():
    """A company moving its primary listing (e.g. NYSE to Nasdaq) should have
    its exchange field updated and the move logged, not silently frozen at
    whatever exchange it was on when first tracked."""
    universe = make_universe([
        dict(ticker="AMD", name="Advanced Micro Devices Common Stock", exchange="NYSE",
             financial_status="N", status="active", first_seen=YESTERDAY, last_seen=YESTERDAY),
    ])
    listed = make_listed([("AMD", "Advanced Micro Devices Common Stock", "NASDAQ", "N")])
    log = u.EventLog(make_events())

    universe, _ = u.sync_universe(TODAY, listed, universe, log)
    events = log.finalize()

    row = universe[universe["ticker"] == "AMD"].iloc[0]
    assert row["exchange"] == "NASDAQ"

    change_events = events[(events["ticker"] == "AMD") & (events["event"] == "EXCHANGE_CHANGE")]
    assert len(change_events) == 1
    assert change_events.iloc[0]["detail"] == "Moved from NYSE to NASDAQ"


def test_no_exchange_change_event_when_exchange_is_unchanged():
    universe = make_universe([
        dict(ticker="WMT", name="Walmart Inc Common Stock", exchange="NYSE",
             financial_status="N", status="active", first_seen=YESTERDAY, last_seen=YESTERDAY),
    ])
    listed = make_listed([("WMT", "Walmart Inc Common Stock", "NYSE", "N")])
    log = u.EventLog(make_events())

    u.sync_universe(TODAY, listed, universe, log)
    events = log.finalize()

    assert events[events["ticker"] == "WMT"].empty


def test_financial_status_refreshes_for_tickers_that_stayed_listed():
    universe = make_universe([
        dict(ticker="XYZ", name="XYZ Corp Common Stock", exchange="NASDAQ",
             financial_status="N", status="active", first_seen=YESTERDAY, last_seen=YESTERDAY),
    ])
    listed = make_listed([("XYZ", "XYZ Corp Common Stock", "NASDAQ", "D")])  # now flagged Deficient
    log = u.EventLog(make_events())

    universe, _ = u.sync_universe(TODAY, listed, universe, log)

    assert universe[universe["ticker"] == "XYZ"].iloc[0]["financial_status"] == "D"
