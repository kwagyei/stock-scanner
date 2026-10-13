"""Offline tests for strategies.py (timing, points, blocking) and v0's unchanged scoring."""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import pytest

import news_sources as ns
import scanner
import strategies as st

UTC = dt.UTC
# Mon 5 – Fri 9 Oct 2026, then Tue 13 (Mon 12 treated as a holiday: missing from the price data)
DAYS = pd.DatetimeIndex(["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09", "2026-10-13"])


@pytest.fixture
def cfg():
    return scanner.load_config(scanner.ROOT / "config.yaml")


def utc(s):  # Stockholm is UTC+2 in October (CEST until 25 Oct)
    return pd.Timestamp(s, tz="UTC")


# --------------------------------------------------------------------------- entry timing


@pytest.mark.parametrize(
    "published, entry",
    [
        ("2026-10-07 06:59", "2026-10-07"),  # 08:59 Stockholm -> same day's open
        ("2026-10-07 07:00", "2026-10-08"),  # 09:00 exactly -> next day
        ("2026-10-07 12:00", "2026-10-08"),  # during the session -> next day
        ("2026-10-06 22:30", "2026-10-07"),  # overnight -> next morning
        ("2026-10-09 15:00", "2026-10-13"),  # Friday afternoon -> Tuesday (Monday holiday)
        ("2026-10-10 09:00", "2026-10-13"),  # Saturday -> Tuesday
        ("2026-10-12 06:00", "2026-10-13"),  # holiday morning -> Tuesday
    ],
)
def test_entry_day_for(published, entry):
    assert st.entry_day_for(utc(published), DAYS) == pd.Timestamp(entry)


def test_entry_day_unknown_beyond_data_and_signal_bar():
    assert st.entry_day_for(utc("2026-10-13 08:00"), DAYS) is None
    assert st.signal_bar_for(utc("2026-10-09 15:00"), DAYS) == pd.Timestamp("2026-10-09")
    assert st.signal_bar_for(utc("2026-10-07 06:00"), DAYS) == pd.Timestamp("2026-10-06")
    assert st.signal_bar_for(utc("2026-10-05 06:00"), DAYS) is None  # entry on the first day


def test_entry_day_handles_winter_time():
    days = pd.DatetimeIndex(["2026-11-02", "2026-11-03"])  # CET, UTC+1
    assert st.entry_day_for(utc("2026-11-02 07:30"), days) == pd.Timestamp("2026-11-02")  # 08:30 local
    assert st.entry_day_for(utc("2026-11-02 08:00"), days) == pd.Timestamp("2026-11-03")  # 09:00 local


def rel(rid, ticker, published, title="Order", source_type="primary"):
    return {
        "release_id": rid,
        "ticker": ticker,
        "source": "mfn",
        "source_type": source_type,
        "published_at": published,
        "title": title,
        "url": rid,
        "language": "sv",
        "body_text": "",
        "fetched_at": published,
    }


def frame(*rows):
    return pd.DataFrame(list(rows), columns=ns.RELEASE_COLUMNS)


def test_releases_for_bar_window():
    r = frame(
        rel("before", "A", "2026-10-08T06:30:00Z"),  # 08:30 Thu -> Thu's open, bar Wed
        rel("in", "A", "2026-10-08T07:30:00Z"),  # 09:30 Thu -> bar Thu
        rel("night", "A", "2026-10-08T20:00:00Z"),  # 22:00 Thu -> bar Thu
        rel("other", "B", "2026-10-08T07:30:00Z"),
        rel("agg", "A", "2026-10-08T07:30:00Z", source_type="aggregator"),
    )
    evening = dt.datetime(2026, 10, 8, 16, 52, tzinfo=UTC)
    assert list(st.releases_for_bar(r, "A", "2026-10-08", evening)["release_id"]) == ["in"]
    morning = dt.datetime(2026, 10, 9, 6, 7, tzinfo=UTC)  # Fri 08:07, bar still Thu
    assert list(st.releases_for_bar(r, "A", "2026-10-08", morning, False)["release_id"]) == ["in", "night"]


def test_releases_during_a_running_session_wait_for_that_days_bar():
    r = frame(rel("fri", "A", "2026-10-09T08:00:00Z"))  # Fri 10:00
    midday = dt.datetime(2026, 10, 9, 9, 0, tzinfo=UTC)  # Fri 11:00, bar = Thu
    assert st.releases_for_bar(r, "A", "2026-10-08", midday, bar_complete_today=False).empty
    # Holiday Monday evening: no Monday bar, so a Monday release belongs to Friday's bar
    r = frame(rel("mon", "A", "2026-10-12T08:00:00Z"))
    mon_evening = dt.datetime(2026, 10, 12, 16, 52, tzinfo=UTC)
    assert list(st.releases_for_bar(r, "A", "2026-10-09", mon_evening, True)["release_id"]) == ["mon"]


# --------------------------------------------------------------------------- points and blocking


def score(rid, direction, material=True):
    return {
        "release_id": rid,
        "direction": str(direction),
        "material": str(material),
        "status": "ok",
        "prompt_version": "v1",
        "is_backfill": "false",
    }


@pytest.mark.parametrize(
    "direction, material, pts",
    [
        (2, True, 3),
        (2, False, 0),
        (1, True, 2),
        (1, False, 2),
        (0, True, 0),
        (-1, True, 0),
        (-2, True, 0),
    ],
)
def test_llm_points_mapping(cfg, direction, material, pts):
    assert st.llm_points(score("r", direction, material), cfg) == pts


def test_block_threshold_and_mixed_releases(cfg):
    bar = frame(rel("good", "A", "2026-10-08T07:30:00Z"), rel("bad", "A", "2026-10-08T08:00:00Z"))
    s = pd.DataFrame([score("good", 2), score("bad", -2)])
    out = st.v1_release(bar, s, cfg)
    assert out["points"] == 3 and out["block"] is True  # most positive -> points, most negative -> block
    out = st.v1_release(bar, pd.DataFrame([score("good", 1), score("bad", -2, material=False)]), cfg)
    assert out == {"points": 2, "block": False, "scored_ids": ["good", "bad"]}
    out = st.v1_release(bar, pd.DataFrame([score("bad", -1)]), cfg)
    assert out["block"] is False  # -1 is above the -2 threshold
    assert st.v1_release(bar, pd.DataFrame(columns=["release_id"]), cfg)["points"] == 0


def test_strategy_scores_replace_release_component(cfg):
    pts = {"release": 2, "volume": 1, "price": 1, "news": 0, "small_cap": 1}  # v0 = 5 (via Google News wire)
    row = {"ticker": "A", "date": "2026-10-08", "points": pts, "market_cap": 1e9}
    now = dt.datetime(2026, 10, 8, 16, 52, tzinfo=UTC)
    none = st.strategy_scores(row, frame(), pd.DataFrame(columns=["release_id"]), cfg, now)
    assert (none["score_v0"], none["score_v0b"], none["score_v1"]) == (5, 3, 3)
    r = frame(rel("in", "A", "2026-10-08T07:30:00Z"))
    got = st.strategy_scores(row, r, pd.DataFrame([score("in", 2)]), cfg, now)
    assert (got["score_v0b"], got["score_v1"], got["block_buy"]) == (5, 6, False)
    assert got["release_ids"] == ["in"] and got["llm_release_ids"] == ["in"]
    # only a release: without it there's no signal, so no small-cap point either
    bare = {"ticker": "A", "date": "2026-10-08", "points": dict.fromkeys(pts, 0), "market_cap": 1e9}
    got = st.strategy_scores(bare, r, pd.DataFrame([score("in", -2)]), cfg, now)
    assert (got["score_v0b"], got["score_v1"], got["block_buy"]) == (3, 0, True)


def test_routine_releases_do_not_count(cfg):
    row = {
        "ticker": "A",
        "date": "2026-10-08",
        "points": dict.fromkeys(scanner.SCORE_COMPONENTS, 0),
        "market_cap": None,
    }
    r = frame(rel("r", "A", "2026-10-08T07:30:00Z", title="Antalet aktier och röster i A AB"))
    now = dt.datetime(2026, 10, 8, 16, 52, tzinfo=UTC)
    assert st.strategy_scores(row, r, pd.DataFrame(columns=["release_id"]), cfg, now)["score_v0b"] == 0


def test_strategy_labels(cfg):
    assert st.strategy_labels(cfg) == ["v0", "v0b"]  # llm.enabled: false (free)
    on = {**cfg, "llm": {**cfg["llm"], "enabled": True}}
    assert st.strategy_labels(on) == ["v0", "v0b", "v1_v1"]
    assert st.llm_enabled(on) and not st.llm_enabled(cfg) and not st.llm_enabled({})


# --------------------------------------------------------------------------- v0 unchanged (golden)

NOW = dt.datetime(2026, 10, 6, 17, 0, tzinfo=UTC)
GOLDEN = [  # (chg_1d, vol_ratio, chg_5d, release sources, n news 24h, mcap) -> (points, hype)
    ((0.12, 6.0, 0.15, ["MFN"], 0, 1e9), ({"release": 2, "volume": 2, "price": 1, "news": 0, "small_cap": 1}, False)),
    ((0.02, 1.2, 0.60, [], 0, 1e9), ({"release": 0, "volume": 0, "price": 0, "news": 0, "small_cap": 0}, True)),
    ((-0.11, 3.5, -0.2, [], 0, 9e9), ({"release": 0, "volume": 1, "price": 1, "news": 0, "small_cap": 0}, False)),
    (
        (0.00, math.nan, 0.0, ["Cision News"], 0, None),
        ({"release": 2, "volume": 0, "price": 0, "news": 0, "small_cap": 0}, False),
    ),
]


@pytest.mark.parametrize("inputs, expected", GOLDEN)
def test_v0_scoring_golden(cfg, inputs, expected):
    chg1, vr, chg5, sources, _, mcap = inputs
    pm = {
        "date": "2026-10-06",
        "close": 10.0,
        "chg_1d": chg1,
        "chg_5d": chg5,
        "volume": 1,
        "vol_ratio": vr,
        "turnover_sek": 10.0,
    }
    items = [
        {"title": "Bolaget tecknar avtal", "link": f"l{i}", "source": s, "ts": NOW - dt.timedelta(hours=2)}
        for i, s in enumerate(sources)
    ]
    nm = scanner.news_metrics(items, NOW, cfg)
    pts, _, hype = scanner.score_breakdown(pm, nm, mcap, cfg)
    assert (pts, hype) == expected


# --------------------------------------------------------------------------- v0b entry filters


def _book(*trades):
    cols = ["ticker", "exit_reason", "exit_date"]
    return pd.DataFrame([dict(zip(cols, t)) for t in trades], columns=cols)


def test_move_filter_skips_extreme_signal_days():
    f = {"max_signal_move": 0.30}
    assert not st.entry_allowed({"ticker": "A", "date": "2026-10-12", "chg_1d": 0.45}, f, _book())
    assert not st.entry_allowed({"ticker": "A", "date": "2026-10-12", "chg_1d": -0.31}, f, _book())
    assert st.entry_allowed({"ticker": "A", "date": "2026-10-12", "chg_1d": 0.30}, f, _book())  # at the limit
    assert st.entry_allowed({"ticker": "A", "date": "2026-10-12", "chg_1d": None}, f, _book())  # unknown move


def test_cooldown_after_a_stop_counts_weekdays():
    f = {"cooldown_days_after_stop": 10}
    stopped = _book(("FASTAT", "stop", "2026-10-08"))

    def row(d, t="FASTAT"):
        return {"ticker": t, "date": d, "chg_1d": -0.05}

    assert not st.entry_allowed(row("2026-10-08"), f, stopped)  # same day
    assert not st.entry_allowed(row("2026-10-09"), f, stopped)  # the real re-entry from week one
    assert not st.entry_allowed(row("2026-10-21"), f, stopped)  # 9 weekdays later
    assert st.entry_allowed(row("2026-10-22"), f, stopped)  # 10 weekdays later
    assert st.entry_allowed(row("2026-10-09", "OTHER"), f, stopped)  # another stock
    other_exits = _book(("FASTAT", "target", "2026-10-08"), ("FASTAT", "time", "2026-10-08"))
    assert st.entry_allowed(row("2026-10-09"), f, other_exits)  # only stop-losses count
    assert st.entry_allowed(row("2026-10-09"), {}, stopped)  # no filters configured


def test_v0b_applies_the_filters_and_v0_does_not(cfg):
    import scanner

    paper = scanner.load_paper().iloc[0:0]
    stops = []
    for s in ("v0", "v0b"):  # both books were stopped out of FASTAT yesterday
        rec = {c: "" for c in scanner.PAPER_COLUMNS}
        rec.update(
            id=f"{s}:FASTAT.ST-2026-10-07",
            ticker="FASTAT.ST",
            strategy=s,
            status="closed",
            exit_reason="stop",
            exit_date="2026-10-08",
            signal_date="2026-10-07",
        )
        stops.append(rec)
    paper = pd.concat([paper, pd.DataFrame(stops)], ignore_index=True)
    rows = [
        {"ticker": "FASTAT.ST", "name": "Fastator", "date": "2026-10-08", "score": 5, "score_v0b": 5, "chg_1d": 0.02},
        {"ticker": "ONCO.ST", "name": "Oncopeptides", "date": "2026-10-08", "score": 4, "score_v0b": 4, "chg_1d": 0.45},
        {"ticker": "CHEF.ST", "name": "Cheffelo", "date": "2026-10-08", "score": 7, "score_v0b": 5, "chg_1d": -0.12},
    ]
    for s in ("v0", "v0b"):
        paper = scanner.add_paper_signals(paper, rows, cfg, "v2", strategy=s)
    pending = paper[paper["status"] == "pending"]
    got = {s: sorted(pending.loc[pending["strategy"] == s, "ticker"]) for s in ("v0", "v0b")}
    assert got == {"v0": ["CHEF.ST", "FASTAT.ST", "ONCO.ST"], "v0b": ["CHEF.ST"]}
