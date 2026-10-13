"""Offline tests for scanner.py. All fetchers are mocked; sockets are blocked."""

from __future__ import annotations

import datetime as dt
import json
import socket

import pandas as pd
import pytest

import evaluate as ev
import scanner

UTC = dt.UTC
NOW = dt.datetime(2026, 10, 6, 17, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def guard(*a, **k):
        raise RuntimeError("network access in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


@pytest.fixture
def cfg():
    # The real config: tests assume the rules are left fixed.
    return scanner.load_config(scanner.ROOT / "config.yaml")


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(scanner, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(scanner, "SIGNALS_LOG", tmp_path / "data" / "signals_log.csv")
    monkeypatch.setattr(scanner, "PAPER_LOG", tmp_path / "data" / "paper_trades.csv")
    monkeypatch.setattr(scanner, "WEEKLY_LOG", tmp_path / "data" / "weekly_summary.csv")
    monkeypatch.setattr(scanner, "SCORES_LOG", tmp_path / "data" / "scores_log.csv")
    monkeypatch.setattr(scanner, "RELEASES_LOG", tmp_path / "data" / "releases.csv")
    monkeypatch.setattr(scanner, "LLM_SCORES_LOG", tmp_path / "data" / "llm_scores.csv")
    monkeypatch.setattr(scanner, "LLM_EVENTS_LOG", tmp_path / "data" / "llm_events.csv")
    monkeypatch.setattr(scanner.ls, "PROMPT_REGISTRY", tmp_path / "used_prompts.json")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(scanner, "DASHBOARD", tmp_path / "DASHBOARD.md")
    monkeypatch.setattr(scanner, "FUND_SUMMARY", tmp_path / "fund_watch_latest.json")
    return tmp_path


def bars(rows, start="2026-06-01"):
    """rows: list of (open, high, low, close, volume) -> daily OHLCV on business days."""
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])


def flat(n, px=100.0, vol=1000, start="2026-06-01"):
    return bars([(px, px, px, px, vol)] * n, start=start)


def news(source, hours_ago, title="t"):
    return {"title": title, "link": "http://x", "source": source, "ts": NOW - dt.timedelta(hours=hours_ago)}


# --------------------------------------------------------------------------- rules guard


def test_rules_unchanged(cfg):
    pc, t = cfg["paper_trading"], cfg["thresholds"]
    assert (pc["min_score"], pc["take_profit"], pc["stop_loss"], pc["max_hold_days"]) == (4, 0.30, -0.15, 20)
    assert (t["volume_mild"], t["volume_strong"], t["price_move_1d"]) == (3, 5, 0.10)
    assert t["small_cap_max_sek"] == 5_000_000_000 and t["hype_5d_move"] == 0.50


# --------------------------------------------------------------------------- scoring


def test_quiet_stock_scores_zero_even_if_small_cap(cfg):
    pm = scanner.price_metrics(flat(70), cfg)
    nm = scanner.news_metrics([], NOW, cfg)
    score, reasons, hype = scanner.score_stock(pm, nm, 1e8, cfg)
    assert (score, reasons, hype) == (0, [], False)


def test_all_signals_fire(cfg):
    h = flat(70)
    h.iloc[-1] = (100, 115, 100, 112, 6000)  # +12%, 6x volume
    pm = scanner.price_metrics(h, cfg)
    items = [news("Cision News", 2), news("di.se", 3), news("EFN", 5)]
    nm = scanner.news_metrics(items, NOW, cfg)
    score, reasons, _ = scanner.score_stock(pm, nm, 1e9, cfg)
    assert score == 2 + 2 + 1 + 1 + 1
    assert reasons[0] == "company release <24h"
    assert "small cap" in reasons


@pytest.mark.parametrize("vol,expected", [(2900, 0), (3000, 1), (4999, 1), (5000, 2)])
def test_volume_tiers(cfg, vol, expected):
    h = flat(70)
    h.iloc[-1, h.columns.get_loc("Volume")] = vol
    pm = scanner.price_metrics(h, cfg)
    score, _, _ = scanner.score_stock(pm, scanner.news_metrics([], NOW, cfg), None, cfg)
    assert score == expected


@pytest.mark.parametrize("mcap", [None, 0, 5e9, 2e10])
def test_small_cap_bonus_needs_known_cap_below_threshold(cfg, mcap):
    h = flat(70)
    h.iloc[-1, h.columns.get_loc("Close")] = 89  # -11% → +1
    pm = scanner.price_metrics(h, cfg)
    score, _, _ = scanner.score_stock(pm, scanner.news_metrics([], NOW, cfg), mcap, cfg)
    assert score == 1


def test_release_older_than_24h_scores_nothing_but_counts_for_window(cfg):
    nm = scanner.news_metrics([news("storage.mfn.se", 48)], NOW, cfg)
    assert nm["releases_24h"] == [] and nm["has_release_window"]


def test_news_spike_needs_count_and_multiple(cfg):
    recent = [news("di.se", h) for h in (1, 2, 3)]
    assert scanner.news_metrics(recent, NOW, cfg)["spike"]
    # 6 older items over 6 days = 1/day baseline; 3 is still >= 3x
    older = [news("di.se", 24 * d + 1) for d in range(1, 7)]
    assert scanner.news_metrics(recent + older, NOW, cfg)["spike"]
    # 12 older items = 2/day baseline; 3 < 6 → no spike
    assert not scanner.news_metrics(recent + older + older, NOW, cfg)["spike"]
    assert not scanner.news_metrics(recent[:2], NOW, cfg)["spike"]


# --------------------------------------------------------------------------- hype warning


def _run_up(pct):
    h = flat(70)
    h.iloc[-1, h.columns.get_loc("Close")] = 100 * (1 + pct)
    return h


@pytest.mark.parametrize(
    "pct,items,expected",
    [
        (0.50, [], True),
        (0.49, [], False),
        (0.80, [news("Cision News", 100)], False),  # release in window explains it
        (0.80, [news("Placera.se", 2)], True),  # media coverage is not a release
    ],
)
def test_hype_warning(cfg, pct, items, expected):
    pm = scanner.price_metrics(_run_up(pct), cfg)
    _, _, hype = scanner.score_stock(pm, scanner.news_metrics(items, NOW, cfg), None, cfg)
    assert hype is expected


# --------------------------------------------------------------------------- missing tickers


def test_missing_and_short_history_tickers_reported(cfg, tmp_paths, monkeypatch):
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query\nGOOD.ST,Good,Good\nBAD.ST,Bad,Bad\nNEW.ST,New,New\n")
    data = {"GOOD.ST": flat(70), "BAD.ST": pd.DataFrame(), "NEW.ST": flat(10), "^OMX": flat(70)}
    news_calls = []
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: data.get(t, pd.DataFrame()))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: news_calls.append(q) or [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: None)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)

    assert scanner.main(["--no-email", "--watchlist", str(wl)]) == 0

    html = (tmp_paths / "reports" / "latest.html").read_text()
    assert "No price data for: BAD.ST, NEW.ST" in html
    assert news_calls == ["Good"]  # no news requests for missing tickers


# --------------------------------------------------------------------------- paper trades


def pending_trade(ticker="X.ST", signal_date="2026-06-01"):
    rec = {c: "" for c in scanner.PAPER_COLUMNS}
    rec.update(
        id=f"{ticker}-{signal_date}",
        ticker=ticker,
        name="X",
        signal_date=signal_date,
        score="4",
        status="pending",
        universe="v1",
    )
    return pd.DataFrame([rec], columns=scanner.PAPER_COLUMNS)


def run_trade(cfg, rows, bench=None):
    """Signal on day 0 (first bar); rows are the bars from the entry day on."""
    h = bars([(100, 100, 100, 100, 1000)] + rows)
    bench = bench if bench is not None else flat(len(h), px=1000)
    return scanner.update_paper(pending_trade(), {"X.ST": h}, bench, cfg).iloc[0]


def test_pending_stays_pending_without_next_bar(cfg):
    p = scanner.update_paper(pending_trade(), {"X.ST": flat(1)}, flat(1), cfg)
    assert p.iloc[0]["status"] == "pending" and p.iloc[0]["entry_price"] == ""


def test_pending_to_open_to_closed_across_runs(cfg):
    full = bars(
        [
            (100, 100, 100, 100, 1000),  # signal day
            (102, 104, 101, 103, 1000),  # entry at open 102
            (103, 105, 100, 104, 1000),
            (104, 140, 103, 135, 1000),
        ]
    )  # target 132.6 hit intraday
    bench = bars(
        [
            (1000, 1000, 1000, 1000, 0),
            (1010, 1010, 1010, 1010, 0),
            (1010, 1010, 1010, 1020, 0),
            (1030, 1030, 1030, 1050, 0),
        ]
    )
    p = pending_trade()

    p = scanner.update_paper(p, {"X.ST": full.iloc[:1]}, bench.iloc[:1], cfg)
    assert p.iloc[0]["status"] == "pending"

    p = scanner.update_paper(p, {"X.ST": full.iloc[:3]}, bench.iloc[:3], cfg)
    r = p.iloc[0]
    assert (r["status"], r["entry_date"], r["entry_price"]) == ("open", "2026-06-02", "102.0000")
    assert r["bench_entry"] == "1010.0000" and r["last_price"] == "104.0000"
    assert r["return_pct"] == f"{(104 / 102 - 1) * 100:.2f}"

    p = scanner.update_paper(p, {"X.ST": full}, bench, cfg)
    r = p.iloc[0]
    assert (r["status"], r["exit_reason"], r["exit_date"]) == ("closed", "target", "2026-06-04")
    assert float(r["exit_price"]) == pytest.approx(102 * 1.3, abs=1e-4)
    assert r["return_pct"] == "30.00"
    assert r["bench_return_pct"] == f"{(1050 / 1010 - 1) * 100:.2f}"


def test_target_gap_up_fills_at_open(cfg):
    r = run_trade(cfg, [(100, 101, 99, 100, 1), (140, 145, 138, 142, 1)])
    assert (r["exit_reason"], r["exit_price"]) == ("target", "140.0000")


def test_stop_intraday(cfg):
    r = run_trade(cfg, [(100, 101, 99, 100, 1), (98, 99, 84, 90, 1)])
    assert (r["exit_reason"], r["exit_price"], r["return_pct"]) == ("stop", "85.0000", "-15.00")


def test_stop_gap_down_fills_at_open(cfg):
    r = run_trade(cfg, [(100, 101, 99, 100, 1), (80, 82, 78, 79, 1)])
    assert (r["exit_reason"], r["exit_price"], r["return_pct"]) == ("stop", "80.0000", "-20.00")


def test_stop_on_entry_day_fills_at_stop_not_open(cfg):
    r = run_trade(cfg, [(100, 101, 84, 86, 1)])
    assert (r["exit_reason"], r["exit_price"], r["exit_date"]) == ("stop", "85.0000", "2026-06-02")


def test_stop_wins_when_bar_hits_both(cfg):
    r = run_trade(cfg, [(100, 101, 99, 100, 1), (100, 140, 80, 120, 1)])
    assert r["exit_reason"] == "stop"


def test_time_exit_on_20th_bar_at_close(cfg):
    rows = [(100, 105, 95, 100 + i * 0.1, 1) for i in range(25)]
    r = run_trade(cfg, rows)
    entry_dates = pd.bdate_range("2026-06-02", periods=20)
    assert (r["exit_reason"], r["exit_date"]) == ("time", entry_dates[-1].date().isoformat())
    assert r["exit_price"] == f"{100 + 19 * 0.1:.4f}"


def test_open_before_time_limit(cfg):
    r = run_trade(cfg, [(100, 105, 95, 101, 1)] * 19)
    assert r["status"] == "open" and r["exit_reason"] == ""


def test_add_signals_skips_active_and_low_scores(cfg):
    p = pending_trade("A.ST")
    rows = [
        {"ticker": "A.ST", "name": "A", "date": "2026-06-02", "score": 6},
        {"ticker": "B.ST", "name": "B", "date": "2026-06-02", "score": 4},
        {"ticker": "C.ST", "name": "C", "date": "2026-06-02", "score": 3},
    ]
    p = scanner.add_paper_signals(p, rows, cfg)
    assert list(p["ticker"]) == ["A.ST", "B.ST"]
    assert p.iloc[1]["status"] == "pending" and p.iloc[1]["id"] == "B.ST-2026-06-02"


# --------------------------------------------------------------------------- CSV round-trip


def test_paper_csv_round_trip(cfg, tmp_paths):
    scanner.DATA_DIR.mkdir()
    open_trade = scanner.update_paper(pending_trade("O.ST"), {"O.ST": flat(3)}, flat(3, px=1000), cfg)
    closed = run_trade(cfg, [(100, 101, 99, 100, 1), (80, 82, 78, 79, 1)]).to_frame().T
    df = pd.concat([open_trade, closed, pending_trade("P.ST")], ignore_index=True)

    df.to_csv(scanner.PAPER_LOG, index=False)
    back = scanner.load_paper()
    pd.testing.assert_frame_equal(back, df.astype(str).assign(strategy="v0"), check_dtype=False)
    assert list(back["status"]) == ["open", "closed", "pending"]

    # Re-running the update on loaded string data with the same prices changes nothing
    # (benchmarks are recomputed on every run, from the same index data).
    again = scanner.update_paper(back.copy(), {"X.ST": flat(3)}, flat(3, px=1000), cfg)
    pd.testing.assert_series_equal(again.iloc[1], back.iloc[1])


def test_load_paper_adds_missing_columns(tmp_paths):
    scanner.DATA_DIR.mkdir()
    scanner.PAPER_LOG.write_text("id,ticker,status\nX-1,X.ST,pending\n")
    p = scanner.load_paper()
    assert list(p.columns) == scanner.PAPER_COLUMNS
    assert p.iloc[0]["status"] == "pending" and p.iloc[0]["entry_price"] == ""


def test_main_persists_trade_lifecycle(cfg, tmp_paths, monkeypatch):
    """End to end through main(): signal → pending row on disk → closed on a later run."""
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query\nX.ST,X,X\n")
    h = flat(70)
    h.iloc[-1] = (100, 112, 100, 111, 6000)  # +11%, 6x vol, small cap → 4
    data = {"X.ST": h, "^OMX": flat(70, px=1000), "nasdaq:IX3364": flat(70, px=500)}
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: data[t])
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: 1e9)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)

    scanner.main(["--no-email", "--watchlist", str(wl)])
    p = pd.read_csv(scanner.PAPER_LOG, dtype=str).fillna("")
    assert list(p[["ticker", "status", "score"]].iloc[0]) == ["X.ST", "pending", "4"]

    nxt = pd.bdate_range(h.index[-1] + pd.Timedelta(days=1), periods=2)
    data["X.ST"] = pd.concat(
        [h, pd.DataFrame([(111, 112, 110, 111, 1000), (90, 91, 80, 85, 1000)], index=nxt, columns=h.columns)]
    )
    data["^OMX"] = flat(72, px=1000)
    scanner.main(["--no-email", "--watchlist", str(wl)])
    p = pd.read_csv(scanner.PAPER_LOG, dtype=str).fillna("")
    assert sorted(p["strategy"]) == ["v0", "v0b"]  # default config: LLM off, no v1 book
    p = p[p["strategy"] == "v0"]
    assert len(p) == 1
    assert list(p[["status", "exit_reason", "exit_price"]].iloc[0]) == ["closed", "stop", "90.0000"]


# --------------------------------------------------------------------------- MFN releases

MFN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel xmlns:x="https://mfn.se/schemas/rss-ns-x/"><title>MFN</title>
<item><title>Eltel vinner ramavtal</title><link>https://mfn.se/a/eltel/sv-1</link>
  <pubDate>Mon, 06 Oct 2026 06:00:00 +0000</pubDate><x:language>sv</x:language></item>
<item><title>Eltel wins frame agreement</title><link>https://mfn.se/a/eltel/en-1</link>
  <pubDate>Mon, 06 Oct 2026 06:00:00 +0000</pubDate><x:language>en</x:language></item>
<item><title>Half-year report</title><link>https://mfn.se/cis/a/eltel/en-2</link>
  <pubDate>Tue, 21 Jul 2026 06:00:00 +0000</pubDate><x:language>en</x:language></item>
</channel></rss>"""

EMPTY_XML = """<?xml version="1.0"?><rss version="2.0"><channel><title>MFN</title></channel></rss>"""


def mock_feed(monkeypatch, xml):
    real_parse, urls = scanner.feedparser.parse, []
    monkeypatch.setattr(scanner.feedparser, "parse", lambda url, agent=None: urls.append(url) or real_parse(xml))
    return urls


def test_fetch_releases_dedupes_language_twins(cfg, monkeypatch):
    urls = mock_feed(monkeypatch, MFN_XML)
    rel = scanner.fetch_releases("eltel", cfg)
    assert urls == ["https://mfn.se/all/a/eltel.rss?limit=20"]
    assert [r["title"] for r in rel] == ["Eltel vinner ramavtal", "Half-year report"]
    assert rel[0]["ts"] == dt.datetime(2026, 10, 6, 6, 0, tzinfo=UTC)
    assert {r["source"] for r in rel} == {"MFN"}


def test_fetch_releases_warns_on_unknown_slug(cfg, monkeypatch, capsys):
    mock_feed(monkeypatch, EMPTY_XML)
    assert scanner.fetch_releases("no-such-co", cfg) == []
    assert "check mfn_slug" in capsys.readouterr().err


def mfn(hours_ago, title="release"):
    return {"title": title, "link": "http://mfn", "source": "MFN", "ts": NOW - dt.timedelta(hours=hours_ago)}


def test_mfn_release_scores_but_does_not_count_as_news(cfg):
    pm = scanner.price_metrics(flat(70), cfg)
    rels = [mfn(2, "Order"), mfn(3), mfn(4)]
    nm = scanner.news_metrics([news("di.se", 1)], NOW, cfg, rels)
    assert nm["n_24h"] == 1 and not nm["spike"]
    assert nm["headlines"][0]["title"] == "Order"
    score, reasons, _ = scanner.score_stock(pm, nm, None, cfg)
    assert (score, reasons) == (2, ["company release <24h"])


@pytest.mark.parametrize("hours_ago,expected_hype", [(100, False), (24 * 30, True)])
def test_mfn_release_window_for_hype(cfg, hours_ago, expected_hype):
    pm = scanner.price_metrics(_run_up(0.6), cfg)
    nm = scanner.news_metrics([], NOW, cfg, [mfn(hours_ago)])
    assert scanner.score_stock(pm, nm, None, cfg)[2] is expected_hype


MFN_BODY_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:x="https://mfn.se/schemas/rss-ns-x/"><channel>
<item><title>A AB tecknar avtal</title><link>https://mfn.se/a/a-co/avtal</link>
<pubDate>Tue, 06 Oct 2026 06:30:00 +0000</pubDate><x:language>sv</x:language>
<x:content type="text">A AB har tecknat ett avtal.</x:content></item>
<item><title>A AB signs agreement</title><link>https://mfn.se/a/a-co/agreement</link>
<pubDate>Tue, 06 Oct 2026 06:30:00 +0000</pubDate><x:language>en</x:language>
<x:content type="text">A AB has signed an agreement.</x:content></item>
</channel></rss>"""


def test_main_fetches_releases_only_with_slug(cfg, tmp_paths, monkeypatch):
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query,mfn_slug\nA.ST,A,A,a-co\nB.ST,B,B,\n")
    calls = []
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: flat(70))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner.ns, "fetch_mfn_xml", lambda s, n: calls.append(s) or MFN_BODY_XML)
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: None)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    scanner.main(["--no-email", "--watchlist", str(wl)])
    assert calls == ["a-co"]
    stored = scanner.ns.load_releases(scanner.RELEASES_LOG)
    assert list(stored["ticker"]) == ["A.ST"] and stored["source_type"].iloc[0] == "primary"
    scanner.main(["--no-email", "--no-news", "--watchlist", str(wl)])
    assert calls == ["a-co"]


# --------------------------------------------------------------------------- routine releases

ROUTINE = [
    "Antalet aktier och röster i Medivir AB per den 30 september 2026",
    "Förändring av det totala antalet aktier och röster i Sivers Semiconductors AB",
    "Förändring av antal röster i Hexatronic",
    "Number of shares and votes in Medivir AB on September 30",
    "Catena Media announces share buybacks executed from 28 September to 2 October 2026",
    "Emission och återköp av C-aktier för incitamentsprogram",
    "Valberedning utsedd i Smart Eye AB inför årsstämman 2027",
    "Valberedningens förslag till styrelse i MilDef Group",
    "Inbjudan till presentation om MilDef Q2 rapport 2026, den 16 juli",
    "Inbjudan till NCAB:s presentation av tredje kvartalet 2026, fredag den 23 oktober 2026",
    "Ferronordic AB: Inbjudan till Ferronordics investerarpresentation 12 augusti 2026",
    "Invitation to presentation of Catena Media's Q2 results 2026",
]
MATERIAL = [
    "Eltel Norge tilldelas långsiktigt partneravtal med Telenor Norge",
    "Sivers Semiconductors har beslutat om en riktad nyemission av aktier om cirka 700 MSEK",
    "Kallelse till extra bolagsstämma i Sivers Semiconductors AB",
    "Catena Media commences share buyback programme",
    "Notification of Major Holdings in Catena Media plc",
    "MEDIVIR AB – DELÅRSRAPPORT JANUARI – JUNI 2026",
    "Episurfs styrelse har beslutat om nyemission av B-aktier och konvertibler som del av vederlaget",
]


@pytest.mark.parametrize("title", ROUTINE)
def test_routine_release_ignored(cfg, title):
    nm = scanner.news_metrics([], NOW, cfg, [mfn(2, title)])
    assert nm["releases_24h"] == [] and not nm["has_release_window"]


@pytest.mark.parametrize("title", MATERIAL)
def test_material_release_counts(cfg, title):
    nm = scanner.news_metrics([], NOW, cfg, [mfn(2, title)])
    assert len(nm["releases_24h"]) == 1


def test_routine_release_does_not_suppress_hype(cfg):
    pm = scanner.price_metrics(_run_up(0.6), cfg)
    items = [news("Cision News", 3, ROUTINE[0])]
    nm = scanner.news_metrics(items, NOW, cfg, [mfn(5, ROUTINE[4])])
    score, reasons, hype = scanner.score_stock(pm, nm, None, cfg)
    assert "company release <24h" not in reasons and hype
    assert nm["n_24h"] == 1  # still counted as news


# --------------------------------------------------------------------------- weekly summary

FRI_EVE = dt.datetime(2026, 10, 9, 18, 45)  # Friday, ISO week 41


@pytest.mark.parametrize(
    "when,expected",
    [
        (FRI_EVE, True),
        (dt.datetime(2026, 10, 9, 8, 30), False),  # Friday morning run
        (dt.datetime(2026, 10, 8, 18, 45), False),  # Thursday
        (dt.datetime(2026, 10, 10, 12, 0), True),  # weekend manual run, week not yet done
    ],
)
def test_weekly_due_schedule(cfg, tmp_paths, when, expected):
    assert scanner.weekly_due(when, cfg) is expected


def test_weekly_due_once_per_week_unless_forced(cfg, tmp_paths):
    scanner.DATA_DIR.mkdir()
    scanner.WEEKLY_LOG.write_text("week,generated_local\n2026-W41,2026-10-06 20:23\n")
    assert scanner.weekly_due(FRI_EVE, cfg)  # early forced run doesn't count
    scanner.WEEKLY_LOG.write_text("week,generated_local\n2026-W41,2026-10-09 18:46\n")
    assert not scanner.weekly_due(FRI_EVE, cfg)
    assert scanner.weekly_due(FRI_EVE, cfg, force=True)
    assert scanner.weekly_due(dt.datetime(2026, 10, 16, 18, 45), cfg)


def test_load_signals_dedupes_runs_keeping_best_score(tmp_paths):
    scanner.DATA_DIR.mkdir()
    scanner.SIGNALS_LOG.write_text(
        "run_utc,bar_date,ticker,name,score,reasons\n"
        "2026-10-06T06:30,2026-10-05,A.ST,A,3,x\n"
        "2026-10-06T16:45,2026-10-06,A.ST,A,4,y\n"
        "2026-10-07T06:30,2026-10-06,A.ST,A,3,z\n"
        "2026-10-06T16:45,2026-10-06,B.ST,B,1,w\n"
    )
    s = scanner.load_signals()
    assert list(zip(s["bar_date"], s["ticker"], s["score"])) == [
        ("2026-10-05", "A.ST", 3),
        ("2026-10-06", "A.ST", 4),
        ("2026-10-06", "B.ST", 1),
    ]


def _signals(*rows):
    return pd.DataFrame([dict(zip(["bar_date", "ticker", "name", "score"], r)) for r in rows])


def test_signal_outcomes_vs_benchmark(cfg):
    h = bars([(100, 100, 100, c, 1) for c in (100, 110, 105, 120)])
    bench = bars([(1000, 1000, 1000, c, 0) for c in (1000, 1050, 1000, 1000)])
    sig = _signals(("2026-06-01", "X.ST", "X", 4), ("2026-06-03", "X.ST", "X", 3), ("2026-06-01", "GONE.ST", "Gone", 5))
    o = scanner.signal_outcomes(sig, {"X.ST": h}, bench, [1, 2])
    assert o.loc[0, "ret_1d"] == pytest.approx(0.10) and o.loc[0, "excess_1d"] == pytest.approx(0.05)
    assert o.loc[0, "ret_2d"] == pytest.approx(0.05) and o.loc[0, "excess_2d"] == pytest.approx(0.05)
    assert o.loc[1, "ret_1d"] == pytest.approx(120 / 105 - 1)
    assert pd.isna(o.loc[1, "ret_2d"])  # not enough bars yet
    assert o.loc[2, ["ret_1d", "ret_2d"]].isna().all()  # ticker without data


def test_hit_rate_table_buckets():
    o = pd.DataFrame(
        {"score": [3, 3, 4, 6], "ret_5d": [0.10, -0.05, 0.02, 0.30], "excess_5d": [0.08, -0.06, -0.01, float("nan")]}
    )
    t = {r["score"]: r for r in scanner.hit_rate_table(o, [5], top_score=5)}
    assert list(t) == ["all", "3", "4", "5+"]
    assert (t["all"]["signals"], t["all"]["n_5d"]) == (4, 3)
    assert t["all"]["hit_5d"] == pytest.approx(100 / 3)
    assert t["3"]["hit_5d"] == 50 and t["3"]["avg_5d"] == pytest.approx(2.5)
    assert t["5+"]["n_5d"] == 0 and t["5+"]["hit_5d"] is None and t["5+"]["avg_5d"] == pytest.approx(30)


def _trade(ticker, status, exit_date="", ret="", bench=""):
    rec = {c: "" for c in scanner.PAPER_COLUMNS}
    rec.update(ticker=ticker, name=ticker, status=status, exit_date=exit_date, return_pct=ret, bench_return_pct=bench)
    return rec


def test_weekly_paper_week_vs_all_time():
    paper = pd.DataFrame(
        [
            _trade("A", "closed", "2026-09-20", "30.00", "2.00"),
            _trade("B", "closed", "2026-10-06", "-15.00", "1.00"),
            _trade("C", "closed", "2026-10-08", "10.00", "-1.00"),
            _trade("D", "open", ret="5.00"),
            _trade("E", "pending"),
        ],
        columns=scanner.PAPER_COLUMNS,
    )
    wp = scanner.weekly_paper(paper, dt.date(2026, 10, 5), 10000)
    assert wp["week"]["n"] == 2 and wp["week"]["win_rate"] == 50
    assert wp["week"]["avg_return"] == pytest.approx(-2.5) and wp["week"]["pnl_sek"] == pytest.approx(-500)
    assert wp["week"]["avg_excess"] == pytest.approx(-2.5)
    assert wp["all"]["n"] == 3 and wp["all"]["pnl_sek"] == pytest.approx(2500)
    assert wp["all"]["avg_bench"] == pytest.approx(2 / 3)
    assert len(wp["open"]) == 1 and wp["unrealised_sek"] == pytest.approx(500)


def test_main_weekly_writes_report_and_one_row_per_week(cfg, tmp_paths, monkeypatch):
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query\nX.ST,X,X\n")
    start = (pd.Timestamp.today().normalize() - pd.offsets.BDay(69)).date()  # last bar = today
    h = flat(70, start=start)
    h.iloc[-1] = (100, 112, 100, 111, 6000)
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: h if t == "X.ST" else flat(70, 1000, start=start))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: 1e9)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    monkeypatch.setattr(scanner, "today_bar_complete", lambda now, cfg: True)  # as if after the close
    sent = []
    monkeypatch.setattr(scanner, "send_email", lambda html, subject: sent.append(subject))

    scanner.main(["--no-email", "--weekly", "--watchlist", str(wl)])
    scanner.main(["--no-email", "--weekly", "--watchlist", str(wl)])

    log = pd.read_csv(scanner.WEEKLY_LOG, dtype=str)
    assert len(log) == 1 and log.iloc[0]["signals_window"] == "1"
    html = (tmp_paths / "reports" / "weekly_latest.html").read_text()
    assert "Signal hit rate" in html and "X.ST" in html
    assert (tmp_paths / "reports" / f"weekly_{log.iloc[0]['week']}.html").exists()
    assert sent == []  # --no-email respected


# --------------------------------------------------------------------------- watchlist versions


def test_watchlist_version(tmp_path):
    wl = tmp_path / "w.csv"
    wl.write_text("# note\nticker,name\nA.ST,A\n")
    assert scanner.watchlist_version(wl) == "v1"
    wl.write_text("# note\n# version: auto-2026-10-06\nticker,name\nA.ST,A\n")
    assert scanner.watchlist_version(wl) == "auto-2026-10-06"
    assert list(scanner.load_watchlist(wl)["ticker"]) == ["A.ST"]


def test_new_trades_tagged_and_old_trades_backfilled(cfg, tmp_paths):
    rows = [{"ticker": "B.ST", "name": "B", "date": "2026-06-02", "score": 4}]
    p = scanner.add_paper_signals(pending_trade("A.ST"), rows, cfg, "auto-2026-10-06")
    assert list(p["universe"]) == ["v1", "auto-2026-10-06"]
    scanner.DATA_DIR.mkdir()
    scanner.PAPER_LOG.write_text("id,ticker,status\nX-1,X.ST,pending\n")  # pre-versioning file
    assert scanner.load_paper().iloc[0]["universe"] == "v1"


def test_weekly_paper_split_by_universe():
    a = _trade("A", "closed", "2026-09-20", "30.00", "2.00")
    b = _trade("B", "closed", "2026-10-06", "-10.00", "1.00")
    a["universe"], b["universe"] = "v1", "auto-2026-10-06"
    wp = scanner.weekly_paper(pd.DataFrame([a, b], columns=scanner.PAPER_COLUMNS), dt.date(2026, 10, 5), 10000)
    assert wp["by_universe"]["v1"]["pnl_sek"] == pytest.approx(3000)
    assert wp["by_universe"]["auto-2026-10-06"]["avg_return"] == pytest.approx(-10)


# --------------------------------------------------------------------------- title filter


@pytest.mark.parametrize(
    "word,title,keep",
    [
        ("Stille", "Pareto höjer riktkursen för Stille till 295 kronor, upprepar köp", True),
        ("Stille", "Stilles vd köper aktier", True),
        ("Stille", "Stille-aktien rusar", True),
        ("Stille", "Lördagsrysare i arenan: ”Jag gillar verkligen det vi gör”", False),
        ("Stille", "Ein stille Nacht", False),  # case-sensitive
        ("I-Tech", "Strategen: Se möjligheterna utanför tech", False),
        ("I-Tech", "I-Tech tecknar nytt avtal för Selektope", True),
        ("K33", "K330 lanseras", False),
        ("Pricer", "Sifonen ersätter Quaero Capital i valberedningen i Pricer", True),
        ("Pricer", "Pricers vd: starkt kvartal", True),
    ],
)
def test_filter_titles(word, title, keep):
    assert (scanner.filter_titles([news("di.se", 1, title)], word) != []) is keep


def test_filter_titles_blank_keeps_all():
    items = [news("di.se", 1, "anything")]
    assert scanner.filter_titles(items, "") == items


def test_main_applies_title_match(cfg, tmp_paths, monkeypatch):
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query,title_match\nS.ST,Stille,Stille,Stille\nO.ST,Other,Other,\n")
    feed = [
        news("di.se", h, t) for h, t in ((1, "Stille rusar"), (2, "Rockbjörnens röda matta"), (3, "Seger för RIG Haga"))
    ]
    start = pd.bdate_range(end=NOW.date(), periods=70)[0]  # latest bar = NOW's day, as in a real run
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: flat(70, start=start))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: feed)
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: None)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    monkeypatch.setattr(scanner, "_now", lambda: NOW)  # news items are timed relative to NOW
    scanner.main(["--no-email", "--watchlist", str(wl)])
    log = pd.read_csv(scanner.SIGNALS_LOG).set_index("ticker")
    assert "S.ST" not in log.index  # 1 relevant item: no spike, no signal
    assert log.loc["O.ST", "news_24h"] == 3 and "news spike" in log.loc["O.ST", "reasons"]


# --------------------------------------------------------------------------- partial-exit variant
# Entry 100 at the day-1 open: target 130, stop 85, remainder stop 100 (breakeven).

D1 = (100, 101, 99, 100, 1)


def test_partial_config_defaults(cfg):
    px = cfg["paper_trading"]["partial_exit"]
    assert (px["enabled"], px["fraction"], px["remainder_stop"]) == (True, 0.5, 0.0)


def test_partial_then_breakeven(cfg):
    r = run_trade(cfg, [D1, (110, 131, 109, 125, 1), (120, 121, 99, 100, 1)])
    assert (r["status"], r["exit_reason"], r["return_pct"]) == ("closed", "target", "30.00")  # main unchanged
    assert (r["pe_status"], r["pe_partial_date"], r["pe_partial_price"]) == ("closed", "2026-06-03", "130.0000")
    assert (r["pe_exit_reason"], r["pe_exit_price"], r["pe_exit_date"]) == ("remainder stop", "100.0000", "2026-06-04")
    assert r["pe_return_pct"] == "15.00"


def test_partial_then_time_exit(cfg):
    rows = [D1, (110, 131, 109, 125, 1)] + [(135, 138, 132, 135, 1)] * 17 + [(138, 141, 136, 140, 1)]
    r = run_trade(cfg, rows)
    assert (r["pe_exit_reason"], r["pe_exit_price"]) == ("time", "140.0000")
    assert r["pe_return_pct"] == "35.00"  # 0.5 x 30% + 0.5 x 40%


def test_partial_gap_up_fills_at_open(cfg):
    r = run_trade(cfg, [D1, (140, 145, 138, 142, 1), (142, 143, 98, 99, 1)])
    assert r["pe_partial_price"] == "140.0000" and r["pe_return_pct"] == "20.00"


@pytest.mark.parametrize("bar,price", [((98, 99, 84, 90, 1), "85.0000"), ((80, 82, 78, 79, 1), "80.0000")])
def test_stop_before_target_exits_whole_position(cfg, bar, price):
    r = run_trade(cfg, [D1, bar])
    assert (r["pe_exit_reason"], r["pe_exit_price"], r["pe_partial_date"]) == ("stop", price, "")
    assert r["pe_return_pct"] == r["return_pct"]


def test_remainder_gap_down_fills_at_open(cfg):
    r = run_trade(cfg, [D1, (110, 131, 109, 125, 1), (95, 96, 90, 92, 1)])
    assert (r["pe_exit_reason"], r["pe_exit_price"], r["pe_return_pct"]) == ("remainder stop", "95.0000", "12.50")


def test_target_on_last_day_sells_rest_at_close(cfg):
    rows = [(100, 105, 95, 100, 1)] * 19 + [(110, 131, 109, 128, 1)]
    r = run_trade(cfg, rows)
    assert (r["pe_partial_date"] == r["pe_exit_date"]) and r["pe_exit_reason"] == "time"
    assert r["pe_return_pct"] == "29.00"


def test_remainder_stop_not_checked_on_the_partial_bar(cfg):
    r = run_trade(cfg, [D1, (110, 131, 95, 120, 1), (120, 122, 118, 120, 1)])
    assert (r["pe_status"], r["pe_exit_reason"], r["pe_return_pct"]) == ("open", "", "25.00")


def test_still_open_before_target(cfg):
    r = run_trade(cfg, [D1, (100, 110, 99, 108, 1)])
    assert (r["pe_status"], r["pe_partial_date"], r["pe_return_pct"]) == ("open", "", "8.00")


def test_variant_keeps_running_after_main_trade_closes(cfg):
    base = [(100, 100, 100, 100, 1), D1, (110, 131, 109, 125, 1)]
    h1 = bars(base)
    p = scanner.update_paper(pending_trade(), {"X.ST": h1}, flat(len(h1), 1000), cfg)
    r = p.iloc[0]
    assert (r["status"], r["pe_status"]) == ("closed", "open")
    main_cols = [c for c in scanner.PAPER_COLUMNS if not c.startswith("pe_")]
    before = r[main_cols].copy()

    h2 = bars(base + [(126, 127, 99, 101, 1)])
    p = scanner.update_paper(p, {"X.ST": h2}, flat(len(h2), 1000), cfg)
    r = p.iloc[0]
    assert (r["pe_status"], r["pe_exit_reason"], r["pe_return_pct"]) == ("closed", "remainder stop", "15.00")
    pd.testing.assert_series_equal(r[main_cols], before, check_names=False)


def test_variant_index_return_is_weighted_like_the_position(cfg):
    h = bars([(100, 100, 100, 100, 1), D1, (110, 131, 109, 125, 1), (120, 121, 99, 100, 1)])
    bench = bars([(1000, 1000, 1000, c, 0) for c in (1000, 1000, 1100, 1200)])
    bench.iloc[1, 0] = 1000  # entry at the index open
    r = scanner.update_paper(pending_trade(), {"X.ST": h}, bench, cfg).iloc[0]
    assert r["pe_bench_return_pct"] == "15.00"  # 0.5 x 10% + 0.5 x 20%


@pytest.mark.parametrize(
    "rows",
    [
        [D1, (110, 131, 109, 125, 1), (120, 121, 99, 100, 1)],
        [D1, (80, 82, 78, 79, 1)],
        [(100, 105, 95, 100 + i * 0.1, 1) for i in range(25)],
        [D1, (100, 110, 99, 108, 1)],
    ],
)
def test_main_rules_identical_with_variant_on_or_off(cfg, rows):
    import copy

    off = copy.deepcopy(cfg)
    off["paper_trading"]["partial_exit"]["enabled"] = False
    main_cols = [c for c in scanner.PAPER_COLUMNS if not c.startswith("pe_")]
    on_row, off_row = run_trade(cfg, rows), run_trade(off, rows)
    pd.testing.assert_series_equal(on_row[main_cols], off_row[main_cols])
    assert off_row["pe_status"] == ""


def test_variant_summaries(cfg):
    a = _trade("A", "closed", "2026-10-06", "30.00", "1.00")
    a.update(pe_status="closed", pe_exit_date="2026-10-07", pe_return_pct="15.00", pe_bench_return_pct="2.00")
    b = _trade("B", "closed", "2026-09-01", "-15.00", "0.00")
    b.update(pe_status="closed", pe_exit_date="2026-09-01", pe_return_pct="-15.00", pe_bench_return_pct="0.00")
    paper = pd.DataFrame([a, b], columns=scanner.PAPER_COLUMNS)
    s = scanner.paper_summary(paper)
    assert (s["pe_n_closed"], s["pe_avg_return"], s["pe_avg_bench"]) == (2, 0.0, 1.0)
    wp = scanner.weekly_paper(paper, dt.date(2026, 10, 5), 10000)
    assert wp["pe_week"]["n"] == 1 and wp["pe_week"]["pnl_sek"] == pytest.approx(1500)
    assert wp["pe_all"]["pnl_sek"] == pytest.approx(0)


def test_open_variant_ticker_is_still_fetched(cfg, tmp_paths, monkeypatch):
    scanner.DATA_DIR.mkdir()
    t = _trade("OLD.ST", "closed", "2026-10-01", "30.00")
    t.update(entry_date="2026-09-01", entry_price="100", pe_status="open")
    pd.DataFrame([t], columns=scanner.PAPER_COLUMNS).to_csv(scanner.PAPER_LOG, index=False)
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query\nX.ST,X,X\n")
    fetched = []
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: fetched.append(t) or flat(70))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: None)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    scanner.main(["--no-email", "--watchlist", str(wl)])
    assert "OLD.ST" in fetched


# --------------------------------------------------------------------------- markdown pages


def _row(**k):
    r = {
        "ticker": "A.ST",
        "name": "A | Co",
        "score": 6,
        "close": 10.0,
        "chg_1d": 0.12,
        "chg_5d": 0.2,
        "vol_ratio": 6.0,
        "n_24h": 2,
        "hype": False,
        "date": "2026-10-06",
        "reasons": ["company release <24h", "volume 6.0x avg", "price +12% 1d", "small cap"],
        "points": {"release": 2, "volume": 2, "price": 1, "news": 0, "small_cap": 1},
        "market_cap": 1.5e9,
        "n_releases": 1,
        "headlines": [{"title": "A | Co wins order", "link": "https://x.test/a?b=(1)"}],
    }
    r.update(k)
    return r


@pytest.mark.parametrize(
    "raw,nice",
    [
        ("company release <24h", "📣 Company release"),
        ("volume 5.1x avg", "📊 Volume 5.1× normal"),
        ("price -11% 1d", "📉 Big drop ▼ 11%"),
        ("price +10% 1d", "📈 Big jump ▲ 10%"),
        ("news spike (3 in 24h vs 1.0/day)", "📰 3 news items today (usually 1.0/day)"),
        ("small cap", "🔹 Small cap"),
        ("something new", "something new"),
    ],
)
def test_friendly_reasons(raw, nice):
    assert scanner._friendly(raw) == nice


def test_price_reason_without_number_next_to_a_price_column():
    assert scanner._friendly("price -11% 1d", short_price=True) == "📉 Big drop"
    assert "📈 Big jump ·" in scanner._alerts_table([_row()], "")


def test_day_format():
    assert scanner._day("2026-10-06") == "Tue&nbsp;6&nbsp;Oct" and scanner._day("junk") == "junk"


def test_why_accepts_list_or_logged_string():
    assert scanner._why("volume 5.1x avg; small cap") == "📊 Volume 5.1× normal · 🔹 Small cap"
    assert scanner._why([]) == "–"


@pytest.mark.parametrize(
    "pct,out", [(2.5, "▲ 2.5%"), (-11.2, "▼ 11.2%"), (0.01, "0.0%"), (None, "–"), (float("nan"), "–")]
)
def test_arrow(pct, out):
    assert scanner._arrow(pct) == out


def test_alert_table_shows_component_values():
    t = scanner._alerts_table([_row()], "")
    head, sep, row = t.splitlines()[:3]
    assert "📣<br><sub>Releases 24h</sub>" in head and "🔹<br><sub>Market cap</sub>" in head
    assert "Today" not in head  # price column replaces it
    # bold = counted: release, volume, price, small cap; news (2 items) did not score
    assert "| **6** | **1** | **6.0×** | **▲&nbsp;12.0%** | 2 | **1.5&nbsp;bn** |" in row
    assert sep.count("|---:") == 5 and "**Bold** = counted towards the score" in t
    assert scanner._alerts_table([], "✅ None.") == "_✅ None._\n" and scanner._alerts_table([], "") == ""


@pytest.mark.parametrize("mc,out", [(None, "–"), (0, "–"), (4.97e8, "497&nbsp;M"), (2.868e9, "2.9&nbsp;bn")])
def test_market_cap_format(mc, out):
    assert scanner._mcap(mc) == out


def test_component_values_without_data():
    v = scanner._component_values(_row(vol_ratio=float("nan"), n_24h=0, n_releases=0, market_cap=None))
    assert (v["volume"], v["news"], v["release"], v["small_cap"]) == ("–", "–", "–", "–")


def test_alert_table_escapes_and_headlines_link():
    t = scanner._alerts_table([_row()], "")
    assert "**A \\| Co**<br><sub>A.ST</sub>" in t and "▲&nbsp;12.0%" in t and "📣 Company release" in t
    h = scanner._headlines([_row()])
    assert h.startswith("<details>") and "](<https://x.test/a?b=(1)>)" in h and "A \\| Co wins order" in h


def test_nav_buttons(monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    root, sub = scanner._nav("DASHBOARD.md"), scanner._nav("reports/latest.md")
    assert "badge/Dashboard-d0d7de" in root and "(reports/latest.md)" in root  # current page greyed
    assert "(../DASHBOARD.md)" in sub and "(latest.md)" not in sub and "(../reports/latest.md)" in sub
    assert "badge/Paper%20trades-" in root and "Runs" not in root
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "me/repo")
    assert "(https://github.com/me/repo/actions)" in scanner._nav("DASHBOARD.md")


def _paper_fixture():
    o = _trade("O.ST", "open", ret="4.00", bench="1.00")
    o.update(entry_date="2026-10-01", entry_price="100.0000", last_price="104.0000", pe_status="open")
    c = _trade("C.ST", "closed", "2026-10-02", "30.00", "1.00")
    c.update(
        entry_date="2026-09-20",
        exit_reason="target",
        pe_status="closed",
        pe_return_pct="15.00",
        pe_exit_reason="remainder stop",
    )
    p = _trade("P.ST", "pending")
    p.update(signal_date="2026-10-06", score="5")
    return pd.DataFrame([o, c, p], columns=scanner.PAPER_COLUMNS)


def test_dashboard(cfg, tmp_paths, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    scanner.DATA_DIR.mkdir()
    scanner.REPORT_DIR.mkdir()
    scanner.SIGNALS_LOG.write_text(
        "run_utc,bar_date,ticker,name,score,reasons\n"
        "2026-10-06T16:45,2026-10-06,A.ST,A,4,volume 6.0x avg\n"
        "2026-10-06T16:45,2026-10-06,B.ST,B,1,small cap\n"
    )
    scanner.WEEKLY_LOG.write_text(
        "week,signals_week,hit_5d,n_5d,hit_20d,n_20d,paper_all_n,paper_all_pnl_sek\n2026-W41,3,50.0,2,,0,1,1500\n"
    )
    (scanner.REPORT_DIR / "weekly_2026-W41.md").write_text("x")
    res = ev.evaluate(ev.closed_trades(_paper_fixture()))
    md = scanner.build_dashboard(
        [_row(), _row(ticker="H.ST", score=0, hype=True)],
        ["BAD.ST"],
        _paper_fixture(),
        cfg,
        dt.datetime(2026, 10, 7, 18, 45),
        res,
    )
    for s in [
        "# 📊 Attention scanner",
        "> [!WARNING]\n> No price data for: BAD.ST",
        "| 🔔 Alerts | ⚠️ Hype warnings | 💼 Open trades | ✅ Closed trades | ⚖️ Evaluation |",
        "| **1**<br><sub>score ≥ 3</sub> | **1** | **1**<br><sub>+1 waiting</sub> | **1** | **NOT YET**",
        "## ⚠️ Hype warnings",
        "> [!CAUTION]",
        "🎯 130.00<br>🛑 85.00",
        "| 5 of 20 |",
        "100.00 → 104.00",
        "⏳ buys at the next open",
        "> [!NOTE]\n> **Evaluation: NOT YET – 1 of 30 trades**",
        "[2026-W41](reports/weekly_2026-W41.md)",
        "50% <sub>of 2</sub>",
        "+1 500 SEK",
        "(reports/paper_trades.md)",
        "(reports/evaluation.md)",
        "(data/signals_log.csv)",
    ]:
        assert s in md, s
    recent = md.split("Signals in the last 10 trading days")[1].split("</details>")[0]
    assert "A.ST" in recent and "B.ST" not in recent  # score 1 isn't a signal
    assert md.count("<details>") == md.count("</details>")


def test_dashboard_empty(cfg, tmp_paths, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    md = scanner.build_dashboard(
        [], [], pd.DataFrame(columns=scanner.PAPER_COLUMNS), cfg, dt.datetime(2026, 10, 6, 8, 30), ev.evaluate(None)
    )
    for s in ["✅ No alerts on this run.", "No signals yet.", "first one comes on Friday", "No open positions."]:
        assert s in md, s
    assert "## ⚠️ Hype warnings" not in md and "Waiting to buy" not in md


def test_paper_page(cfg):
    paper = _paper_fixture()
    md = scanner.build_paper_md(paper, cfg, dt.datetime(2026, 10, 7, 18, 45), ev.evaluate(ev.closed_trades(paper)))
    for s in [
        "# 💼 Paper trades",
        "| **Main rules** | 2 | 1 | 100% | ▲ 30.0% | ▲ 29.0% | +3 000 SEK |",
        "| **Partial-exit variant** | 1 | 1 |",
        "## Open (1)",
        "## Waiting to buy (1)",
        "## Closed (1)",
        "2026-09-20 → 2026-10-02",
        "🎯 Target",
        "▲ 15.0%<br><sub>🛑 Breakeven stop</sub>",
    ]:
        assert s in md, s


def _eval_res(tmp_path, excess, n):
    rows = [
        {
            "ticker": f"T{i}",
            "status": "closed",
            "return_pct": excess[i % len(excess)] + 1,
            "bench_return_pct": 0,
            "exit_reason": "time",
            "exit_date": f"2026-01-{i % 28 + 1:02d}",
        }
        for i in range(n)
    ]
    path = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return ev.evaluate(ev.load_trades(path))


def test_evaluation_page_verdicts(cfg, tmp_path):
    now = dt.datetime(2026, 10, 7, 18, 45)
    passed = scanner.build_evaluation_md(_eval_res(tmp_path, [1, 5], 40), None, cfg, now)
    assert (
        "> [!TIP]\n> **Evaluation: PASS – 40 trades" in passed
        and "| ✅ | Enough trades | 40 closed (need 30) |" in passed
    )
    failed = scanner.build_evaluation_md(_eval_res(tmp_path, [6, -4.5], 40), None, cfg, now)
    assert "> [!WARNING]" in failed and "Failed: unlikely to be luck (t > 2)." in failed and "| ❌ |" in failed
    empty = scanner.build_evaluation_md(ev.evaluate(None), ev.evaluate(None), cfg, now)
    assert "No paper-trade log yet" in empty and "## Partial-exit variant" in empty
    early = scanner.build_evaluation_md(_eval_res(tmp_path, [1, 5], 5), None, cfg, now)
    assert "| ⏳ | Enough trades | 5 closed (need 30) |" in early and "### By exit reason" in early


def test_main_writes_all_pages(cfg, tmp_paths, monkeypatch):
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query\nX.ST,X,X\n")
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: flat(70))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: None)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    scanner.main(["--no-email", "--weekly", "--watchlist", str(wl)])
    rep = tmp_paths / "reports"
    titles = {
        scanner.DASHBOARD: "# 📊",
        rep / "latest.md": "# 📰 Today",
        rep / "paper_trades.md": "# 💼",
        rep / "evaluation.md": "# ⚖️ Evaluation",
        rep / "weekly_latest.md": "# 📅 Weekly summary",
    }
    for path, title in titles.items():
        text = path.read_text()
        assert text.startswith(title), path
        assert "img.shields.io/badge/Dashboard" in text, path  # button row on every page
        assert text.count("<details>") == text.count("</details>"), path
    assert len(list(rep.glob("weekly_20*.md"))) == 1


def test_weekly_with_no_signals_in_window(cfg, tmp_paths):
    scanner.DATA_DIR.mkdir()
    scanner.REPORT_DIR.mkdir()
    html, md = scanner.run_weekly(pd.DataFrame(columns=scanner.PAPER_COLUMNS), {}, flat(5), cfg, FRI_EVE)
    assert "No signals in the window yet." in md and "No signals this week." in md
    assert "img.shields.io" not in md  # job-summary version has no button row
    assert "img.shields.io" in (scanner.REPORT_DIR / "weekly_latest.md").read_text()


# --------------------------------------------------------------------------- score components / score log


def _scenario(vol=1000, close=100, items=(), mcap=None):
    h = flat(70)
    h.iloc[-1] = (100, max(100, close), min(100, close), close, vol)
    return (
        scanner.price_metrics(h, CFG_FOR_SCENARIOS),
        scanner.news_metrics(list(items), NOW, CFG_FOR_SCENARIOS),
        mcap,
    )


CFG_FOR_SCENARIOS = scanner.load_config(scanner.ROOT / "config.yaml")


@pytest.mark.parametrize(
    "kw,expected",
    [
        ({}, {}),
        ({"vol": 3500}, {"volume": 1}),
        ({"vol": 6000, "close": 112, "mcap": 1e9}, {"volume": 2, "price": 1, "small_cap": 1}),
        ({"mcap": 1e9}, {}),  # small cap alone: nothing
        (
            {"items": [news("Cision News", 2), news("di.se", 3), news("EFN", 5)], "mcap": 1e9},
            {"release": 2, "news": 1, "small_cap": 1},
        ),
        ({"close": 85, "mcap": 9e9}, {"price": 1}),  # big cap: no small-cap point
    ],
)
def test_score_breakdown_components_sum_to_score(cfg, kw, expected):
    pm, nm, mcap = _scenario(**kw)
    pts, reasons, hype = scanner.score_breakdown(pm, nm, mcap, cfg)
    assert pts == {**dict.fromkeys(scanner.SCORE_COMPONENTS, 0), **expected}
    assert scanner.score_stock(pm, nm, mcap, cfg) == (sum(pts.values()), reasons, hype)


def test_main_logs_every_stock_with_components(cfg, tmp_paths, monkeypatch):
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("# version: auto-test\nticker,name,query\nHOT.ST,Hot,Hot\nQUIET.ST,Quiet,Quiet\nGONE.ST,Gone,Gone\n")
    hot = flat(70)
    hot.iloc[-1] = (100, 112, 100, 111, 6000)
    data = {"HOT.ST": hot, "QUIET.ST": flat(70), "^OMX": flat(70, 1000)}
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: data.get(t, pd.DataFrame()))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: 1.5e9)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)

    scanner.main(["--no-email", "--watchlist", str(wl)])
    log = pd.read_csv(scanner.SCORES_LOG, dtype=str).fillna("").set_index("ticker")
    assert list(log.index) == ["HOT.ST", "QUIET.ST", "GONE.ST"]
    h = log.loc["HOT.ST"]
    assert (h["score"], h["pts_volume"], h["pts_price"], h["pts_small_cap"], h["pts_release"], h["pts_news"]) == (
        "4",
        "2",
        "1",
        "1",
        "0",
        "0",
    )
    assert (h["status"], h["market_cap_msek"], h["universe"], h["news_24h"]) == ("ok", "1500", "auto-test", "0")
    q = log.loc["QUIET.ST"]
    assert (q["score"], q["pts_small_cap"], q["reasons"]) == ("0", "0", "")  # logged although quiet
    g = log.loc["GONE.ST"]
    assert (g["status"], g["score"], g["name"]) == ("no price data", "", "Gone")
    sig = pd.read_csv(scanner.SIGNALS_LOG)
    assert list(sig["ticker"]) == ["HOT.ST"]  # signals log unchanged

    scanner.main(["--no-email", "--no-news", "--watchlist", str(wl)])
    log = pd.read_csv(scanner.SCORES_LOG, dtype=str).fillna("")
    assert len(log) == 6 and log["run_utc"].nunique() >= 1
    assert log.iloc[3]["news_24h"] == "" and log.iloc[3]["releases_24h"] == ""  # news not checked


def test_append_csv_rewrites_when_columns_change(tmp_path):
    path = tmp_path / "log.csv"
    scanner.append_csv(path, [{"a": 1, "b": 2}])
    scanner.append_csv(path, [{"a": 3, "b": 4}])
    scanner.append_csv(path, [{"a": 5, "b": 6, "c": 7}])
    df = pd.read_csv(path, dtype=str).fillna("")
    assert list(df.columns) == ["a", "b", "c"] and df["c"].tolist() == ["", "", "7"]
    scanner.append_csv(path, [])
    assert len(pd.read_csv(path)) == 3


def test_stale_bar_is_skipped_not_backdated(cfg, tmp_paths, monkeypatch):
    """Yahoo dropped the latest bar (seen around midnight): no back-dated signal or paper trade."""
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query\nHOT.ST,Hot,Hot\nOK.ST,Ok,Ok\n")
    hot = flat(70)
    hot.iloc[-1] = (100, 112, 100, 111, 6000)  # would score 4 (with small cap)
    last = hot.index[-1].date().isoformat()
    later = (hot.index[-1] + pd.offsets.BDay(1)).date().isoformat()
    scanner.DATA_DIR.mkdir()
    scanner.SCORES_LOG.write_text(
        f"run_utc,bar_date,ticker,status\nx,{later},HOT.ST,ok\nx,{last},OK.ST,ok\nx,{later},OK.ST,no price data\n"
    )
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: hot if t == "HOT.ST" else flat(70))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: 1e9)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    scanner.main(["--no-email", "--watchlist", str(wl)])

    assert not scanner.PAPER_LOG.exists() or "HOT.ST" not in scanner.PAPER_LOG.read_text()
    assert not scanner.SIGNALS_LOG.exists() or "HOT.ST" not in scanner.SIGNALS_LOG.read_text()
    log = pd.read_csv(scanner.SCORES_LOG, dtype=str).fillna("")
    new = log[log["run_utc"] != "x"].set_index("ticker")
    assert new.loc["HOT.ST", "status"] == "stale price data"
    assert new.loc["OK.ST", "status"] == "ok"  # same date as before is fine
    assert "HOT.ST (stale)" in scanner.DASHBOARD.read_text()


@pytest.mark.parametrize(
    "local_hm,uses_today", [((12, 45), False), ((17, 59), False), ((18, 0), True), ((8, 7), False)]
)
def test_todays_forming_bar_is_ignored_before_cutoff(cfg, tmp_paths, monkeypatch, local_hm, uses_today):
    today = dt.date(2026, 10, 7)
    idx = pd.bdate_range(end=today, periods=70)
    h = pd.DataFrame([(100, 100, 100, 100, 1000)] * 70, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])
    h.iloc[-1] = (100, 112, 100, 111, 6000)  # today's (forming) bar would score 4
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query\nX.ST,X,X\n")
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: h)
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: 1e9)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    local = dt.datetime(2026, 10, 7, *local_hm, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    monkeypatch.setattr(scanner, "_now", lambda: local.astimezone(dt.UTC))
    scanner.main(["--no-email", "--watchlist", str(wl)])
    log = pd.read_csv(scanner.SCORES_LOG, dtype=str).iloc[0]
    assert (log["bar_date"] == "2026-10-07") is uses_today
    traded = scanner.PAPER_LOG.exists() and "X.ST" in scanner.PAPER_LOG.read_text()
    assert traded is uses_today


def test_completed_bars_keeps_entry_day_open_for_pending_trades(cfg):
    # A pending trade still opens at today's open once today's bar counts (evening run).
    h = bars([(100, 100, 100, 100, 1), (102, 104, 101, 103, 1)], start="2026-10-06")
    midday = scanner.completed_bars(h, dt.datetime(2026, 10, 7, 12, 0))
    assert list(midday.index.date) == [dt.date(2026, 10, 6)]
    p = scanner.update_paper(pending_trade(signal_date="2026-10-06"), {"X.ST": midday}, midday, cfg)
    assert p.iloc[0]["status"] == "pending"
    p = scanner.update_paper(p, {"X.ST": h}, h, cfg)
    assert (p.iloc[0]["status"], p.iloc[0]["entry_price"]) == ("open", "102.0000")


# --------------------------------------------------------------------------- direction split (scanner)


def test_new_trade_records_signal_move(cfg):
    p = scanner.add_paper_signals(
        pd.DataFrame(columns=scanner.PAPER_COLUMNS),
        [{"ticker": "E.ST", "name": "E", "date": "2026-10-06", "score": 4, "chg_1d": -0.1117}],
        cfg,
    )
    assert p.iloc[0]["signal_move"] == "-0.1117"


def test_backfill_signal_moves_from_logs_then_history(tmp_paths):
    scanner.DATA_DIR.mkdir()
    scanner.SIGNALS_LOG.write_text("run_utc,bar_date,ticker,name,score,chg_1d\nx,2026-10-06,A.ST,A,4,-0.1117\n")
    scanner.SCORES_LOG.write_text("run_utc,bar_date,ticker,status,chg_1d\nx,2026-10-06,B.ST,ok,0.1049\n")
    trades = [
        pending_trade("A.ST", "2026-10-06"),
        pending_trade("B.ST", "2026-10-06"),
        pending_trade("C.ST", "2026-06-02"),
        pending_trade("D.ST", "2026-06-02"),
    ]
    paper = pd.concat(trades, ignore_index=True)
    paper.loc[3, "signal_move"] = "0.0500"  # already known: untouched
    hist = bars([(100, 100, 100, 100, 1), (90, 90, 90, 90, 1)])  # C: -10% on 2026-06-02
    out = scanner.backfill_signal_moves(paper, {"C.ST": hist, "D.ST": hist})
    assert out["signal_move"].tolist() == ["-0.1117", "0.1049", "-0.1000", "0.0500"]


def test_direction_rate_table(cfg):
    o = pd.DataFrame(
        {
            "score": [3, 4, 3],
            "move": ["-0.12", "0.15", "0.01"],
            "ret_5d": [0.10, -0.05, 0.02],
            "excess_5d": [0.08, -0.06, -0.01],
        }
    )
    t = {r["score"]: r for r in scanner.direction_rate_table(o, [5], 0.10)}
    assert list(t) == ["▼ Drop", "▲ Jump", "– No big move"]
    assert (t["▼ Drop"]["hit_5d"], t["▲ Jump"]["hit_5d"], t["– No big move"]["signals"]) == (100.0, 0.0, 1)


def test_paper_page_shows_signal_day_and_split(cfg):
    paper = _paper_fixture()
    paper.loc[paper["ticker"] == "C.ST", "signal_move"] = "-0.1117"
    paper.loc[paper["ticker"] == "O.ST", "signal_move"] = "0.0300"
    md = scanner.build_paper_md(paper, cfg, dt.datetime(2026, 10, 7, 18, 45), ev.evaluate(ev.closed_trades(paper)))
    assert "| Stock | Signal day |" in md and "▼&nbsp;11.2%" in md and "▲&nbsp;3.0%" in md
    assert "### By signal-day move" in md and "| ▼ Drop | 1 | 100% |" in md


def test_weekly_page_has_direction_split(cfg):
    o = pd.DataFrame(
        {
            "ticker": ["A"],
            "name": ["A"],
            "bar_date": ["2026-09-01"],
            "score": [4],
            "reasons": [""],
            "move": ["-0.12"],
            "ret_1d": [0.01],
            "excess_1d": [0.01],
            "ret_5d": [0.02],
            "excess_5d": [0.02],
            "ret_20d": [None],
            "excess_20d": [None],
        }
    )
    table = scanner.hit_rate_table(o, [1, 5, 20], 5)
    wp = scanner.weekly_paper(pd.DataFrame(columns=scanner.PAPER_COLUMNS), dt.date(2026, 10, 5), 10000)
    md = scanner.build_weekly_md(o, table, wp, cfg, FRI_EVE)
    assert "Split by signal-day move" in md and "| ▼ Drop | 1 |" in md
    assert md.count("<details>") == md.count("</details>")


# --------------------------------------------------------------------------- fund watch on the dashboard


def _fund_summary(run_utc="2026-10-07T05:13+00:00", flagged=2, high=1, single=False):
    top = [
        {
            "score": 6,
            "severity": "HIGH",
            "title": "TSMC | halts production",
            "link": "https://x.test/a",
            "source": "Reuters",
            "entities": ["TSMC"],
            "categories": ["infrastructure"],
            "similar": 3,
            "similar_sources": "Bloomberg, Nikkei Asia, Focus Taiwan",
            "direction": "negative",
        },
        {
            "score": 3,
            "severity": "MEDIUM",
            "title": "Quanta revenue falls",
            "link": "https://x.test/b",
            "source": "digitimes",
            "entities": ["Quanta Computer"],
            "categories": ["company_results"],
            "status": "escalated",
            "first_seen": "2026-10-05T05:13+00:00",
        },
    ][:flagged]
    tw = {
        "n_flagged": flagged,
        "n_high": high,
        "n_watch": 4,
        "top": top,
        "n_developing": 2,
        **({} if single else {"n_flagged_articles": flagged + (3 if flagged else 0)}),
    }
    if single:  # the format written before several funds
        return {"run_utc": run_utc, "run_local": "2026-10-07 07:13", "n_raw": 1300, **tw}
    gl = {
        "id": "avanza_global",
        "short": "Global",
        "icon": "🌍",
        "name": "Avanza Global",
        "isin": "SE0011527613",
        "n_flagged": 1,
        "n_high": 0,
        "n_watch": 9,
        "top": [
            {
                "score": 4,
                "severity": "MEDIUM",
                "title": "Fed raises interest rates",
                "link": "https://x.test/c",
                "source": "Forbes",
                "entities": [],
                "categories": ["macro"],
            }
        ],
    }
    return {
        "run_utc": run_utc,
        "run_local": "2026-10-07 07:13",
        "n_raw": 1300,
        "funds": [
            {"id": "jpm_taiwan", "short": "Taiwan", "icon": "🇹🇼", "name": "JPM Taiwan", **tw},
            gl if flagged else {**gl, "n_flagged": 0, "top": []},
        ],
    }


def _dash(cfg, now=dt.datetime(2026, 10, 7, 18, 52)):
    return scanner.build_dashboard([], [], pd.DataFrame(columns=scanner.PAPER_COLUMNS), cfg, now, ev.evaluate(None))


def test_dashboard_fund_watch_section(cfg, tmp_paths, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    scanner.FUND_SUMMARY.write_text(json.dumps(_fund_summary()))
    md = _dash(cfg)
    glance_head = next(line for line in md.splitlines() if line.startswith("| 🔔"))
    assert "| 🛡️ Fund watch |" in glance_head
    assert "**3**<br><sub>flagged · 1 high</sub>" in md  # summed over both funds
    sec = md.split("## 🛡️ Fund risk watch")[1].split("## More")[0]
    assert "Last run **Wed&nbsp;7&nbsp;Oct 07:13**" in sec and "1300 articles scanned" in sec
    assert "| 🇹🇼 **Taiwan** | **2**<br><sub>5 articles</sub> | 1 | 2 | 4 |" in sec  # ... developing, watch
    assert "| 🌍 **Global** | **1** | 0 | 0 | 9 |" in sec  # no grouping: no article count
    assert "⬆️ **escalated** · first seen Mon&nbsp;5&nbsp;Oct" in sec
    assert "➕ 3 similar: Bloomberg, Nikkei Asia, Focus Taiwan" in sec
    assert "**🇹🇼 Taiwan** · 1 high-severity" in sec and "**🌍 Global**" in sec
    assert (
        "| 🔴&nbsp;HIGH | ▼&nbsp;negative | [TSMC \\| halts production](<https://x.test/a>)"
        "<br><sub>➕ 3 similar: Bloomberg, Nikkei Asia, Focus Taiwan</sub> | TSMC |"
    ) in sec
    assert (
        "| 🟠&nbsp;MEDIUM | ·&nbsp;unclear | [Fed raises interest rates](<https://x.test/c>)" in sec
    )  # older summary: no direction
    assert "[Full fund watch →](fund-watch/reports/fund_watch_latest.md)" in sec
    assert "last ran" not in sec  # fresh: no stale warning


def test_dashboard_reads_old_single_fund_summary(cfg, tmp_paths, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    scanner.FUND_SUMMARY.write_text(json.dumps(_fund_summary(single=True)))
    md = _dash(cfg)
    assert "**2**<br><sub>flagged · 1 high</sub>" in md and "| 🇹🇼 **Taiwan** | **2** | 1 | 0 | 4 |" in md


def test_dashboard_fund_watch_missing_quiet_and_stale(cfg, tmp_paths, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    md = _dash(cfg)
    assert "hasn't run yet" in md and "**–**<br><sub>not run yet</sub>" in md
    scanner.FUND_SUMMARY.write_text(json.dumps(_fund_summary(flagged=0, high=0)))
    assert "✅ Nothing new flagged on the last run. 2 earlier story(ies) still developing." in _dash(cfg)
    stale = _dash(cfg, now=dt.datetime(2026, 10, 10, 9, 0))
    assert "The fund watch last ran 3 day(s) ago" in stale
    scanner.FUND_SUMMARY.write_text("{broken")
    assert "hasn't run yet" in _dash(cfg)


def test_fund_watch_button_on_every_page():
    assert "(fund-watch/reports/fund_watch_latest.md)" in scanner._nav("DASHBOARD.md")
    assert "(../fund-watch/reports/fund_watch_latest.md)" in scanner._nav("reports/latest.md")
    assert "badge/Fund%20watch-" in scanner._nav("reports/evaluation.md")


def test_pages_only_rebuilds_from_logs_without_fetching(cfg, tmp_paths, monkeypatch):
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query\nHOT.ST,Hot,Hot\nGONE.ST,Gone,Gone\n")
    hot = flat(70)
    hot.iloc[-1] = (100, 112, 100, 111, 6000)
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: hot if t == "HOT.ST" else pd.DataFrame())
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: 1.5e9)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    monkeypatch.setattr(scanner, "today_bar_complete", lambda now, cfg: True)
    scanner.main(["--no-email", "--watchlist", str(wl)])
    full = scanner.DASHBOARD.read_text()
    logs = {p: p.read_text() for p in scanner.DATA_DIR.iterdir()}

    def boom(*a, **k):
        raise AssertionError("pages-only must not fetch")

    for name in ("fetch_history", "fetch_news", "fetch_market_cap", "fetch_releases"):
        monkeypatch.setattr(scanner, name, boom)
    scanner.DASHBOARD.unlink()
    assert scanner.main(["--pages-only"]) == 0
    rebuilt = scanner.DASHBOARD.read_text()
    assert {p: p.read_text() for p in scanner.DATA_DIR.iterdir()} == logs  # no log touched
    alerts = lambda md: md.split("## 🔔 Today's alerts")[1].split("## 💼")[0]
    assert alerts(rebuilt) == alerts(full)  # same alert table
    assert "No price data for: GONE.ST" in rebuilt and "badge/Fund%20watch" in rebuilt


def test_pages_only_without_logs_fails_clearly(cfg, tmp_paths):
    with pytest.raises(SystemExit, match="no score log yet"):
        scanner.main(["--pages-only"])


# --------------------------------------------------------------------------- primary benchmark (Nasdaq)

NASDAQ_PAYLOAD = {
    "data": {
        "CP": [
            {
                "z": {
                    "dateTime": "2026-10-06",
                    "open": "1,230.10",
                    "high": "1,240.00",
                    "low": "1,225.00",
                    "close": "1,238.50",
                }
            },
            {
                "z": {
                    "dateTime": "2026-10-07",
                    "open": "1,242.15",
                    "high": "1,245.00",
                    "low": "1,230.00",
                    "close": "1,235.22",
                }
            },
            {"z": {"dateTime": "2026-10-08", "open": "-", "high": "", "low": None, "close": ""}},
        ]
    }
}


def test_parse_nasdaq_chart_handles_comma_numbers_and_blank_rows():
    df = scanner.parse_nasdaq_chart(NASDAQ_PAYLOAD)
    assert list(df.index) == [pd.Timestamp("2026-10-06"), pd.Timestamp("2026-10-07")]
    assert df.loc["2026-10-07", "Open"] == 1242.15
    assert df.loc["2026-10-07", "Close"] == 1235.22
    assert scanner.parse_nasdaq_chart({}).empty


def test_fetch_history_routes_nasdaq_symbols(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(NASDAQ_PAYLOAD).encode()

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        return Resp()

    monkeypatch.setattr(scanner.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(scanner, "_now", lambda: NOW)
    df = scanner.fetch_history("nasdaq:IX3364", "6mo")
    assert len(df) == 2
    assert "/instruments/IX3364/chart" in seen["url"] and "toDate=2026-10-06" in seen["url"]


def test_fetch_nasdaq_index_failure_warns_and_returns_empty(monkeypatch, capsys):
    def boom(*a, **k):
        raise OSError("down")

    monkeypatch.setattr(scanner.urllib.request, "urlopen", boom)
    assert scanner.fetch_nasdaq_index("IX3364", "6mo", today=dt.date(2026, 10, 7)).empty
    assert "Nasdaq index fetch failed" in capsys.readouterr().err


def test_bench_symbol_primary_and_secondary(cfg):
    assert scanner.bench_symbol(cfg, "primary") == "nasdaq:IX3364"
    assert scanner.bench_symbol(cfg) == "^OMX"
    assert scanner.bench_symbol({"benchmark": "^OMX"}, "secondary") == "^OMX"


# --------------------------------------------------------------------------- strategies in the paper book

OLD_PAPER_HEADER = (
    "id,ticker,name,signal_date,score,status,entry_date,entry_price,bench_entry,exit_date,"
    "exit_price,exit_reason,bench_exit,last_price,return_pct,bench_return_pct,universe,pe_status,"
    "pe_partial_date,pe_partial_price,pe_exit_date,pe_exit_price,pe_exit_reason,pe_return_pct,"
    "pe_bench_return_pct,signal_move"
)


def test_paper_migration_marks_existing_rows_v0(tmp_paths):
    scanner.DATA_DIR.mkdir()
    scanner.PAPER_LOG.write_text(
        OLD_PAPER_HEADER + "\nELTEL.ST-2026-10-06,ELTEL.ST,Eltel,2026-10-06,4,open,"
        "2026-10-07,18.0000,2500,,,,,18.0000,0.00,,v2,open,,,,,,0.00,,-0.1200\n"
    )
    p = scanner.load_paper()
    assert list(p.columns) == scanner.PAPER_COLUMNS
    assert p.iloc[0]["strategy"] == "v0" and p.iloc[0]["id"] == "ELTEL.ST-2026-10-06"
    assert p.iloc[0]["bench_primary_entry"] == ""
    assert len(scanner.book(p, "v0")) == 1 and scanner.book(p, "v0b").empty


def strat_row(ticker="X.ST", score=4, v0b=4, v1=5, block=False):
    return {
        "ticker": ticker,
        "name": ticker,
        "date": "2026-10-08",
        "score": score,
        "chg_1d": 0.05,
        "score_v0b": v0b,
        "score_v1": v1,
        "block_buy": block,
    }


def test_three_books_are_independent_and_v1_blocks(cfg, tmp_paths):
    paper = scanner.load_paper()
    rows = [strat_row("A.ST", 4, 2, 2), strat_row("B.ST", 1, 4, 6), strat_row("C.ST", 4, 4, 4, block=True)]
    for s in ("v0", "v0b", "v1_v1"):
        paper = scanner.add_paper_signals(paper, rows, cfg, "v2", strategy=s)
    got = {s: sorted(scanner.book(paper, s)["ticker"]) for s in ("v0", "v0b", "v1_v1")}
    assert got == {"v0": ["A.ST", "C.ST"], "v0b": ["B.ST", "C.ST"], "v1_v1": ["B.ST"]}
    assert set(scanner.book(paper, "v0")["id"]) == {"A.ST-2026-10-08", "C.ST-2026-10-08"}
    assert scanner.book(paper, "v1_v1").iloc[0]["id"] == "v1_v1:B.ST-2026-10-08"
    assert scanner.book(paper, "v1_v1").iloc[0]["score"] == "6"
    # an open v0 trade doesn't stop v0b from opening the same stock
    again = scanner.add_paper_signals(paper, [strat_row("A.ST", 4, 4, 4)], cfg, "v2", strategy="v0b")
    assert len(scanner.book(again, "v0b")) == 3 and len(scanner.book(again, "v0")) == 2


def test_primary_benchmark_return_on_trades(cfg):
    p = scanner.update_paper(
        pending_trade("O.ST"), {"O.ST": flat(3)}, flat(3, px=1000), cfg, bench_primary=flat(3, px=500)
    )
    r = p.iloc[0]
    assert r["status"] == "open" and r["bench_primary_entry"] == "500.0000"
    assert r["bench_primary_return_pct"] == "0.00"


BLOCK_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:x="https://mfn.se/schemas/rss-ns-x/"><channel>
<item><title>X AB utf\xc3\xa4rdar vinstvarning</title><link>https://mfn.se/a/x/warn</link>
<pubDate>Thu, 08 Oct 2026 08:00:00 +0000</pubDate><x:language>sv</x:language>
<x:content type="text">R\xc3\xb6relseresultatet blir betydligt l\xc3\xa4gre \xc3\xa4n v\xc3\xa4ntat.</x:content></item>
</channel></rss>"""


def with_llm(cfg):
    return {**cfg, "llm": {**cfg["llm"], "enabled": True}}


def test_main_end_to_end_llm_blocks_v1_but_not_v0b(cfg, tmp_paths, monkeypatch):
    from types import SimpleNamespace

    import llm_score as ls

    real_load = scanner.load_config
    monkeypatch.setattr(scanner, "load_config", lambda p: with_llm(real_load(p)))
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query,mfn_slug\nX.ST,X,X,x\n")
    start = pd.bdate_range(end="2026-10-08", periods=70)[0]
    h = flat(70, start=start)
    h.iloc[-1] = (100, 101, 85, 88, 6000)  # -12%, 6x volume, small cap
    data = {"X.ST": h, "^OMX": flat(70, px=1000, start=start), "nasdaq:IX3364": flat(70, px=500, start=start)}
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: data[t])
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: 1e9)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    monkeypatch.setattr(scanner.ns, "fetch_mfn_xml", lambda s, n: BLOCK_XML)
    monkeypatch.setattr(scanner, "_now", lambda: dt.datetime(2026, 10, 8, 16, 52, tzinfo=UTC))
    answer = {
        "event_type": "guidance",
        "direction": -2,
        "surprise": 3,
        "material": True,
        "confidence": 0.9,
        "rationale": "Profit warning well below expectations.",
    }
    calls = []

    class Client:
        class messages:
            @staticmethod
            def create(**kw):
                calls.append(kw)
                return SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text=json.dumps(answer))],
                    usage=SimpleNamespace(input_tokens=900, output_tokens=60),
                )

    monkeypatch.setattr(ls, "make_client", lambda: Client)

    scanner.main(["--no-email", "--watchlist", str(wl)])
    assert len(calls) == 1 and "vinstvarning" in calls[0]["messages"][0]["content"]
    scores = ls.load_scores(scanner.LLM_SCORES_LOG)
    assert list(scores[["status", "direction", "is_backfill"]].iloc[0]) == ["ok", "-2", "false"]
    p = pd.read_csv(scanner.PAPER_LOG, dtype=str).fillna("")
    # v0: release 2 (MFN) + volume 2 + price 1 + small cap 1 = 6; v0b same; v1: release 0 -> 4 but blocked
    assert sorted(p["strategy"]) == ["v0", "v0b"]
    sig = pd.read_csv(scanner.SIGNALS_LOG, dtype=str).fillna("")
    r = sig.iloc[0]
    assert (r["score_v0"], r["score_v0b"], r["score_v1"], r["block_buy"]) == ("6", "6", "4", "True")
    assert r["llm_prompt_version"] == "v1" and r["block_if_direction_lte"] == "-2" and r["release_ids"]
    html = (scanner.REPORT_DIR / "latest.html").read_text()
    assert "LLM release scoring" in html and "Profit warning well below expectations." in html
    assert "Blocked v1 buys · 1" in html and "v1_v1" in html
    evm = (scanner.REPORT_DIR / "evaluation.md").read_text()
    assert "## LLM experiment: v0 / v0b / v1" in evm and "| `v0` | `v0b` | `v1_v1` |" in evm
    events = pd.read_csv(scanner.LLM_EVENTS_LOG, dtype=str).fillna("")
    assert list(events["release_id"]) == list(scores["release_id"]) and events.iloc[0]["entry_date"] == ""
    # a second run doesn't call the LLM again
    scanner.main(["--no-email", "--watchlist", str(wl)])
    assert len(calls) == 1


def test_parse_spreads():
    payload = {
        "data": {
            "instrumentListing": {
                "rows": [
                    {"symbol": "4C", "bidPrice": "6.61", "askPrice": "6.72"},
                    {"symbol": "SAAB B", "bidPrice": "1,000.00", "askPrice": "1,001.00"},
                    {"symbol": "NOBID", "bidPrice": "", "askPrice": "3.00"},
                    {"symbol": "CROSSED", "bidPrice": "5.00", "askPrice": "4.00"},
                ]
            }
        }
    }
    sp = scanner.parse_spreads(payload)
    assert set(sp) == {"4C.ST", "SAAB-B.ST"}
    assert sp["4C.ST"] == pytest.approx(0.11 / 6.665 * 100)
    assert scanner.parse_spreads({}) == {}


def test_spread_recorded_on_new_trades(cfg, tmp_paths):
    paper = scanner.add_paper_signals(
        scanner.load_paper().iloc[0:0], [{**strat_row("A.ST"), "spread_pct": 1.6667}], cfg, "v2", strategy="v0b"
    )
    assert paper.iloc[0]["spread_pct"] == "1.667"
    paper = scanner.add_paper_signals(paper, [strat_row("B.ST")], cfg, "v2", strategy="v0b")
    assert paper.iloc[1]["spread_pct"] == ""


def test_llm_failure_becomes_a_report_warning(cfg, tmp_paths, monkeypatch):
    scanner.DATA_DIR.mkdir()

    def boom(*a, **k):
        raise scanner.ls.PromptChanged("prompt v1 changed after it was used live")

    monkeypatch.setattr(scanner.ls, "run", boom)
    rel = scanner.ns.load_releases(scanner.RELEASES_LOG)
    rel.loc[0] = ["r", "X.ST", "mfn", "primary", "2026-10-09T07:00:00Z", "t", "u", "sv", "", ""]
    warnings = []
    scores, new = scanner.score_releases(
        rel, pd.DataFrame({"ticker": ["X.ST"], "name": ["X"]}), with_llm(cfg), warnings
    )
    assert new == [] and warnings == ["LLM scoring failed: prompt v1 changed after it was used live"]
    html = scanner.build_llm_html([], [], rel, None, warnings, with_llm(cfg))
    assert "LLM scoring failed" in html and "class=warn" in html
    md = scanner.build_evaluation_md(
        scanner.ev.evaluate(None), None, cfg, NOW, {"strategies": {}, "events": scanner.ev.event_study(None)}, warnings
    )
    assert "[!WARNING]" in md and "LLM scoring failed" in md


def test_llm_off_by_default_makes_no_api_call(cfg, tmp_paths, monkeypatch):
    """The default config is the free setup: no client is ever created, even with a key set."""
    assert cfg["llm"]["enabled"] is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")

    def no_client():
        raise AssertionError("LLM client created while llm.enabled is false")

    monkeypatch.setattr(scanner.ls, "make_client", no_client)
    scanner.DATA_DIR.mkdir()
    rel = scanner.ns.load_releases(scanner.RELEASES_LOG)
    rel.loc[0] = ["r", "X.ST", "mfn", "primary", "2026-10-09T07:00:00Z", "Order", "u", "sv", "", ""]
    warnings = []
    scores, new = scanner.score_releases(rel, pd.DataFrame({"ticker": ["X.ST"], "name": ["X"]}), cfg, warnings)
    assert new == [] and warnings == []
    html = scanner.build_llm_html([], [], rel, None, [], cfg)
    assert "LLM scoring is off" in html and "Blocked v1 buys" not in html


def test_fill_entries_at_open_during_session_then_evening_run_matches(cfg):
    signal = flat(3)  # signal on the last complete bar
    today = pd.bdate_range(signal.index[-1] + pd.Timedelta(days=1), periods=1)
    forming = pd.concat([signal, pd.DataFrame([(104, 105, 80, 81, 500)], index=today, columns=signal.columns)])
    bench_forming = pd.concat(
        [flat(3, px=1000), pd.DataFrame([(1010, 1010, 1000, 1005, 0)], index=today, columns=signal.columns)]
    )
    p = pending_trade("X.ST", signal.index[-1].date().isoformat())
    # midday: update on complete bars (nothing to do), then fill at today's open
    p = scanner.update_paper(p, {"X.ST": signal}, flat(3, px=1000), cfg)
    p = scanner.fill_entries_at_open(p, {"X.ST": forming}, bench_forming, bench_forming)
    r = p.iloc[0]
    assert (r["status"], r["entry_price"], r["bench_entry"], r["bench_primary_entry"]) == (
        "open",
        "104.0000",
        "1010.0000",
        "1010.0000",
    )
    assert r["exit_date"] == "" and r["bench_primary_return_pct"] == ""  # today's low (80) not used yet
    # another midday run keeps it open and untouched
    again = scanner.update_paper(p.copy(), {"X.ST": signal}, flat(3, px=1000), cfg, flat(3, px=1000))
    assert again.iloc[0]["status"] == "open" and again.iloc[0]["exit_date"] == ""
    # evening: the complete bar arrives; same result as a trade filled only in the evening
    evening = scanner.update_paper(p.copy(), {"X.ST": forming}, bench_forming, cfg, bench_forming)
    direct = scanner.update_paper(
        pending_trade("X.ST", signal.index[-1].date().isoformat()), {"X.ST": forming}, bench_forming, cfg, bench_forming
    )
    cols = [
        "status",
        "entry_date",
        "entry_price",
        "exit_price",
        "exit_reason",
        "return_pct",
        "bench_entry",
        "bench_return_pct",
        "bench_primary_return_pct",
    ]
    assert list(evening.iloc[0][cols]) == list(direct.iloc[0][cols])
    assert evening.iloc[0]["exit_reason"] == "stop"


def test_fill_entries_skips_without_todays_bar(cfg):
    signal = flat(3)
    p = scanner.fill_entries_at_open(
        pending_trade("X.ST", signal.index[-1].date().isoformat()), {"X.ST": signal}, flat(3, px=1000)
    )
    assert p.iloc[0]["status"] == "pending"


def test_evening_runs_refresh_weekly_page_but_only_friday_sends(cfg, tmp_paths, monkeypatch):
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query\nX.ST,X,X\n")
    start = pd.bdate_range(end="2026-10-07", periods=70)[0]
    h = flat(70, start=start)
    h.iloc[-1] = (100, 112, 100, 111, 6000)
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: h if t == "X.ST" else flat(70, 1000, start=start))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: 1e9)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    sent = []
    monkeypatch.setattr(scanner, "send_email", lambda html, subject: sent.append(subject))
    cfg_mail = {**cfg, "email": {**cfg.get("email", {}), "enabled": True, "send_when": "alerts"}}
    monkeypatch.setattr(scanner, "load_config", lambda p: cfg_mail)
    page = tmp_paths / "reports" / "weekly_latest.md"

    # Wednesday morning (before the close): no refresh
    monkeypatch.setattr(scanner, "_now", lambda: dt.datetime(2026, 10, 7, 6, 7, tzinfo=UTC))
    scanner.main(["--watchlist", str(wl)])
    assert not page.exists()
    # Wednesday evening: page refreshed, no weekly e-mail
    monkeypatch.setattr(scanner, "_now", lambda: dt.datetime(2026, 10, 7, 16, 52, tzinfo=UTC))
    scanner.main(["--watchlist", str(wl)])
    assert page.exists() and "X.ST" in page.read_text()
    assert not any(s.startswith("Scanner weekly") for s in sent)
    # Friday evening: the week's final version, e-mailed
    monkeypatch.setattr(scanner, "_now", lambda: dt.datetime(2026, 10, 9, 16, 52, tzinfo=UTC))
    scanner.main(["--watchlist", str(wl)])
    assert any(s.startswith("Scanner weekly") for s in sent)
    assert len(pd.read_csv(scanner.WEEKLY_LOG, dtype=str)) == 1  # one row per week


# --------------------------------------------------------------------------- Yahoo gaps repaired from Nasdaq


def _yahoo(rows):
    idx = pd.DatetimeIndex(["2026-10-06", "2026-10-07", "2026-10-08"]).tz_localize("Europe/Stockholm")
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])


NQ_SHARE = {
    "data": {
        "CP": [
            {
                "z": {
                    "dateTime": "2026-10-08",
                    "open": "98.00",
                    "high": "101.00",
                    "low": "96.80",
                    "close": "99.80",
                    "volume": "205,140",
                }
            }
        ]
    }
}


def test_missing_yahoo_close_is_filled_from_nasdaq(monkeypatch, capsys):
    y = _yahoo(
        [
            (113.4, 115, 110.6, 111.8, 57007),
            (105.2, 105.4, 95.8, 97.8, 772515),
            (98.0, 101.0, 96.8, float("nan"), 202928),
        ]
    )
    monkeypatch.setattr(scanner.yf, "Ticker", lambda t: type("T", (), {"history": lambda self, **k: y})())
    monkeypatch.setattr(scanner, "NASDAQ_SHARE_IDS", {"CHEF.ST": "TX3775430"})
    urls = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(NQ_SHARE).encode()

    monkeypatch.setattr(scanner.urllib.request, "urlopen", lambda req, timeout: urls.append(req.full_url) or Resp())
    h = scanner.fetch_history("CHEF.ST", "6mo")
    assert h.index[-1] == pd.Timestamp("2026-10-08")
    assert (h.iloc[-1]["Close"], h.iloc[-1]["Volume"]) == (99.8, 205140)
    assert h.iloc[-2]["Close"] == 97.8  # Yahoo rows untouched
    assert "assetClass=SHARES" in urls[0] and "TX3775430" in urls[0] and "fromDate=2026-10-08" in urls[0]
    assert "filled 1 missing Yahoo close" in capsys.readouterr().out


def test_missing_close_without_nasdaq_is_dropped_as_before(monkeypatch, capsys):
    y = _yahoo([(1, 1, 1, 1.0, 1), (1, 1, 1, 1.0, 1), (1, 1, 1, float("nan"), 1)])
    monkeypatch.setattr(scanner.yf, "Ticker", lambda t: type("T", (), {"history": lambda self, **k: y})())
    monkeypatch.setattr(scanner, "NASDAQ_SHARE_IDS", {})
    assert scanner.fetch_history("X.ST", "6mo").index[-1] == pd.Timestamp("2026-10-07")
    monkeypatch.setattr(scanner, "NASDAQ_SHARE_IDS", {"X.ST": "TX1"})

    def down(*a, **k):
        raise OSError("down")

    monkeypatch.setattr(scanner.urllib.request, "urlopen", down)
    assert scanner.fetch_history("X.ST", "6mo").index[-1] == pd.Timestamp("2026-10-07")
    assert "Nasdaq couldn't fill it" in capsys.readouterr().err


def test_parse_share_ids():
    payload = {
        "data": {
            "instrumentListing": {
                "rows": [{"symbol": "SAAB B", "orderbookId": "SSE1"}, {"symbol": "X", "orderbookId": ""}]
            }
        }
    }
    assert scanner.parse_share_ids(payload) == {"SAAB-B.ST": "SSE1"}


# --------------------------------------------------------------------------- no look-ahead in daytime runs


def _mfn_xml(*items):
    body = "".join(
        f"<item><title>{t}</title><link>https://mfn.se/a/x/{i}</link><pubDate>{d}</pubDate>"
        f'<x:language>sv</x:language><x:content type="text">{t}.</x:content></item>'
        for i, (t, d) in enumerate(items)
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><rss version="2.0" '
        f'xmlns:x="https://mfn.se/schemas/rss-ns-x/"><channel>{body}</channel></rss>'
    ).encode()


@pytest.mark.parametrize(
    "release_utc, counts",
    [
        ("Thu, 08 Oct 2026 18:00:00 +0000", True),  # Thu 20:00 Stockholm: before Fri's open
        ("Fri, 09 Oct 2026 08:00:00 +0000", False),  # Fri 10:00: after the open the trade buys at
    ],
)
def test_daytime_run_does_not_credit_yesterdays_bar_with_todays_release(
    cfg, tmp_paths, monkeypatch, release_utc, counts
):
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query,mfn_slug\nX.ST,X,X,x\n")
    start = pd.bdate_range(end="2026-10-08", periods=70)[0]  # last complete bar: Thu 8 Oct
    h = flat(70, start=start)
    h.iloc[-1] = (100, 101, 99, 100, 4000)  # 4x volume: +1, small cap +1; a release would add +2
    data = {"X.ST": h, "^OMX": flat(70, px=1000, start=start), "nasdaq:IX3364": flat(70, px=500, start=start)}
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: data[t])
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: 1e9)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    monkeypatch.setattr(scanner.ns, "fetch_mfn_xml", lambda s, n: _mfn_xml(("X AB tecknar avtal", release_utc)))
    monkeypatch.setattr(scanner, "_now", lambda: dt.datetime(2026, 10, 9, 9, 0, tzinfo=UTC))  # Fri 11:00
    scanner.main(["--no-email", "--watchlist", str(wl)])
    row = pd.read_csv(scanner.SCORES_LOG, dtype=str).fillna("").iloc[-1]
    assert row["bar_date"] == "2026-10-08"
    assert (row["pts_release"] == "2") is counts  # v0
    assert (row["score"] == "4") is counts  # v0: 2 without the release, 4 with it
    assert (row["score_v0b"] == "4") is counts  # v0b: the same release window


def test_news_windows_end_at_the_as_of_time(cfg):
    as_of = dt.datetime(2026, 10, 9, 7, 0, tzinfo=UTC)  # Fri 09:00 Stockholm
    items = [
        {"title": "X tecknar avtal", "link": "a", "source": "Cision", "ts": as_of + dt.timedelta(hours=1)},
        {"title": "X i fokus", "link": "b", "source": "di.se", "ts": as_of + dt.timedelta(hours=1)},
        {"title": "X i fokus igen", "link": "c", "source": "di.se", "ts": as_of - dt.timedelta(hours=23)},
        {"title": "X förra veckan", "link": "d", "source": "di.se", "ts": as_of - dt.timedelta(hours=25)},
    ]
    nm = scanner.news_metrics(items, as_of, cfg)
    assert nm["releases_24h"] == []  # the Cision release came after the open
    assert nm["n_24h"] == 1  # only the item 23h before as_of; the later one doesn't count
    assert [i["link"] for i in nm["headlines"]] == ["c"]
    # a later as-of shifts the 24h window: both items after 09:00 count, the one from 23h before 09:00 drops out
    assert scanner.news_metrics(items, as_of + dt.timedelta(hours=2), cfg)["n_24h"] == 2


def test_no_intraday_fill_before_the_open(cfg, tmp_paths, monkeypatch):
    """A morning run before 09:00 must not buy at a pre-open placeholder row for today."""
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query\nX.ST,X,X\n")
    start = pd.bdate_range(end="2026-10-08", periods=70)[0]
    h = flat(70, start=start)
    today = pd.DataFrame([(55, 55, 55, 55, 0)], index=[pd.Timestamp("2026-10-09")], columns=h.columns)
    data = {
        "X.ST": pd.concat([h, today]),
        "^OMX": flat(70, px=1000, start=start),
        "nasdaq:IX3364": flat(70, px=500, start=start),
    }
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: data[t])
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: 1e9)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    scanner.DATA_DIR.mkdir()
    pending_trade("X.ST", "2026-10-08").to_csv(scanner.PAPER_LOG, index=False)
    monkeypatch.setattr(scanner, "_now", lambda: dt.datetime(2026, 10, 9, 6, 7, tzinfo=UTC))  # Fri 08:07
    scanner.main(["--no-email", "--watchlist", str(wl)])
    p = pd.read_csv(scanner.PAPER_LOG, dtype=str).fillna("")
    assert p.iloc[0]["status"] == "pending" and p.iloc[0]["entry_price"] == ""
    monkeypatch.setattr(scanner, "_now", lambda: dt.datetime(2026, 10, 9, 8, 0, tzinfo=UTC))  # Fri 10:00
    scanner.main(["--no-email", "--watchlist", str(wl)])
    p = pd.read_csv(scanner.PAPER_LOG, dtype=str).fillna("")
    assert p.iloc[0]["status"] == "open" and p.iloc[0]["entry_price"] == "55.0000"


def test_scores_log_records_the_as_of_time(cfg, tmp_paths, monkeypatch):
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query\nX.ST,X,X\n")
    start = pd.bdate_range(end="2026-10-08", periods=70)[0]
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: flat(70, start=start))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: None)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    monkeypatch.setattr(scanner, "_now", lambda: dt.datetime(2026, 10, 9, 9, 0, tzinfo=UTC))  # Fri 11:00
    scanner.main(["--no-email", "--watchlist", str(wl)])
    row = pd.read_csv(scanner.SCORES_LOG, dtype=str).fillna("").iloc[-1]
    assert row["as_of"] == "2026-10-09T09:00+02:00"


# --------------------------------------------------------------------------- benchmarks by exact date


def _bars_on(dates, px):
    """OHLCV on the given dates; px = one price per date (open = close = px)."""
    return pd.DataFrame(
        [(p, p, p, p, 1000) for p in px],
        index=pd.DatetimeIndex(dates),
        columns=["Open", "High", "Low", "Close", "Volume"],
    )


STOCK = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09"], [100, 100, 104])


def _entered(bench, bench_primary=None, cfg=None):
    p = pending_trade("X.ST", "2026-10-07")
    return scanner.update_paper(p, {"X.ST": STOCK}, bench, cfg or CFG_FOR_SCENARIOS, bench_primary).iloc[0]


def test_benchmark_uses_exactly_the_trades_days(cfg):
    bench = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09"], [1000, 1010, 1030])
    r = _entered(bench, bench, cfg)
    assert (r["entry_date"], r["bench_entry"], r["bench_exit"]) == ("2026-10-08", "1010.0000", "1030.0000")
    assert r["bench_return_pct"] == r["bench_primary_return_pct"] == f"{(1030 / 1010 - 1) * 100:.2f}"


def test_a_gap_in_the_index_blanks_instead_of_using_the_day_before(cfg):
    """The reviewer's case: no index price on the entry day. The old lookup took the 7 Oct price
    and reported 0% for a day the index moved."""
    bench = _bars_on(["2026-10-07", "2026-10-09"], [1000, 1030])  # 8 Oct missing inside the history
    r = _entered(bench, cfg=cfg)
    assert r["status"] == "open" and r["bench_entry"] == "" and r["bench_return_pct"] == ""


def test_index_not_published_yet_is_left_for_the_next_run(cfg):
    bench = _bars_on(["2026-10-07", "2026-10-08"], [1000, 1010])  # 9 Oct not out yet
    p = scanner.update_paper(pending_trade("X.ST", "2026-10-07"), {"X.ST": STOCK}, bench, cfg)
    assert p.iloc[0]["bench_return_pct"] == ""  # nothing to compare yet, but not a gap
    later = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09"], [1000, 1010, 1030])
    p = scanner.update_paper(p, {"X.ST": STOCK}, later, cfg)
    assert p.iloc[0]["bench_return_pct"] == f"{(1030 / 1010 - 1) * 100:.2f}"


def test_daytime_fill_without_todays_index_gets_its_entry_later(cfg):
    """Finding #3: a trade bought at today's open before the index printed today kept no index
    entry for good. Now the next run fills it in."""
    forming = _bars_on(["2026-10-07", "2026-10-08"], [100, 100])
    p = scanner.fill_entries_at_open(
        pending_trade("X.ST", "2026-10-07"), {"X.ST": forming}, _bars_on(["2026-10-07"], [1000])
    )
    assert p.iloc[0]["status"] == "open" and p.iloc[0]["bench_entry"] == ""
    bench = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09"], [1000, 1010, 1030])
    r = scanner.update_paper(p, {"X.ST": STOCK}, bench, cfg).iloc[0]
    assert r["bench_entry"] == "1010.0000" and r["bench_return_pct"] != ""


def test_closed_trades_are_repaired_on_later_runs(cfg):
    """A closed trade with a benchmark from the wrong day (the old lookup) is recomputed."""
    bench = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09"], [1000, 1010, 1030])
    rec = {c: "" for c in scanner.PAPER_COLUMNS}
    rec.update(
        id="Y",
        ticker="X.ST",
        name="X",
        signal_date="2026-10-07",
        score="4",
        status="closed",
        entry_date="2026-10-08",
        entry_price="100.0000",
        exit_date="2026-10-09",
        exit_price="104.0000",
        exit_reason="time",
        return_pct="4.00",
        bench_entry="1000.0000",
        bench_exit="1030.0000",
        bench_return_pct="3.00",
        universe="v1",
    )
    p = scanner.update_paper(pd.DataFrame([rec], columns=scanner.PAPER_COLUMNS), {"X.ST": STOCK}, bench, cfg)
    assert (p.iloc[0]["bench_entry"], p.iloc[0]["bench_return_pct"]) == ("1010.0000", f"{(1030 / 1010 - 1) * 100:.2f}")


def test_history_older_than_the_index_data_is_kept(cfg):
    """Days before the index history fetched aren't gaps: keep what was computed back then."""
    bench = _bars_on(["2026-10-09"], [1030])
    rec = {c: "" for c in scanner.PAPER_COLUMNS}
    rec.update(
        id="Y",
        ticker="X.ST",
        status="closed",
        entry_date="2026-10-08",
        exit_date="2026-10-08",
        bench_entry="1010.0000",
        bench_exit="1012.0000",
        bench_return_pct="0.20",
        universe="v1",
    )
    p = scanner.refresh_benchmarks(pd.DataFrame([rec], columns=scanner.PAPER_COLUMNS), {"X.ST": STOCK}, bench)
    assert p.iloc[0]["bench_return_pct"] == "0.20"


def test_partial_exit_index_return_is_weighted_by_exact_dates():
    bench = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09"], [1000, 1010, 1030])
    rec = {c: "" for c in scanner.PAPER_COLUMNS}
    rec.update(
        id="Y",
        ticker="X.ST",
        status="closed",
        entry_date="2026-10-08",
        exit_date="2026-10-09",
        pe_status="closed",
        pe_partial_date="2026-10-08",
        pe_exit_date="2026-10-09",
        universe="v1",
    )
    p = scanner.refresh_benchmarks(
        pd.DataFrame([rec], columns=scanner.PAPER_COLUMNS), {"X.ST": STOCK}, bench, None, 0.5
    )
    expected = 0.5 * (1010 / 1010 - 1) + 0.5 * (1030 / 1010 - 1)
    assert p.iloc[0]["pe_bench_return_pct"] == f"{expected * 100:.2f}"


# --------------------------------------------------------------------------- bad opening prices


@pytest.mark.parametrize("bad_open", [0.0, float("nan")])
def test_entry_skips_an_unusable_open(cfg, bad_open):
    """Finding #6: an Open of 0 crashed the run (division by zero); NaN left the trade stuck."""
    h = STOCK.copy()
    h.loc["2026-10-08", "Open"] = bad_open
    p = scanner.update_paper(pending_trade("X.ST", "2026-10-07"), {"X.ST": h}, _bars_on(["2026-10-09"], [1]), cfg)
    r = p.iloc[0]
    assert (r["status"], r["entry_date"], r["entry_price"]) == ("open", "2026-10-09", "104.0000")


def test_entry_waits_when_no_usable_open_yet(cfg):
    h = _bars_on(["2026-10-07", "2026-10-08"], [100, 0])
    p = scanner.update_paper(pending_trade("X.ST", "2026-10-07"), {"X.ST": h}, h, cfg)
    assert p.iloc[0]["status"] == "pending"


def test_weekly_outcomes_need_the_index_on_both_days():
    sig = pd.DataFrame(
        [{"ticker": "X.ST", "name": "X", "bar_date": "2026-10-07", "score": 4, "reasons": "", "chg_1d": ""}]
    )
    bench = _bars_on(["2026-10-07", "2026-10-09"], [1000, 1030])  # 8 Oct missing
    out = scanner.signal_outcomes(sig, {"X.ST": STOCK}, bench, [1, 2])
    assert pd.isna(out.iloc[0]["excess_1d"]) and out.iloc[0]["excess_2d"] == pytest.approx(0.04 - 0.03)


def test_a_stale_value_over_an_index_gap_is_cleared():
    """A closed trade benchmarked by the old nearest-earlier lookup across a real index gap must
    not keep that number: it's blanked, and the evaluation leaves the trade out."""
    bench = _bars_on(["2026-10-07", "2026-10-09"], [1000, 1030])  # 8 Oct missing inside the history
    rec = {c: "" for c in scanner.PAPER_COLUMNS}
    rec.update(
        id="Y",
        ticker="X.ST",
        status="closed",
        entry_date="2026-10-08",
        exit_date="2026-10-09",
        bench_entry="1000.0000",
        bench_exit="1030.0000",
        bench_return_pct="3.00",
        universe="v1",
    )
    p = scanner.refresh_benchmarks(pd.DataFrame([rec], columns=scanner.PAPER_COLUMNS), {"X.ST": STOCK}, bench)
    assert (p.iloc[0]["bench_entry"], p.iloc[0]["bench_exit"], p.iloc[0]["bench_return_pct"]) == ("", "", "")


def test_trade_closed_today_before_the_index_prints_today_is_not_compared_with_yesterday(cfg):
    """Review of the fix: on the evening run Yahoo may not have today's index close yet. A trade
    stopped out today must not keep yesterday's index figures; it waits for the index."""
    h = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09", "2026-10-12"], [100, 100, 100, 100])
    h.loc["2026-10-12", ["Open", "Low", "Close"]] = (90, 40, 50)  # stopped out on 12 Oct
    bench_to_9th = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09"], [1000, 1010, 1030])
    p = scanner.update_paper(pending_trade("X.ST", "2026-10-07"), {"X.ST": h.iloc[:3]}, bench_to_9th, cfg)
    assert p.iloc[0]["bench_return_pct"] != ""  # open, figures up to the 9th
    p = scanner.update_paper(p, {"X.ST": h}, bench_to_9th, cfg)  # evening of the 12th, index not out yet
    r = p.iloc[0]
    assert (r["status"], r["exit_date"]) == ("closed", "2026-10-12")
    assert r["bench_entry"] == "1010.0000" and r["bench_exit"] == "" and r["bench_return_pct"] == ""
    later = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09", "2026-10-12"], [1000, 1010, 1030, 1100])
    r = scanner.update_paper(p, {"X.ST": h}, later, cfg).iloc[0]
    assert r["bench_return_pct"] == f"{(1100 / 1010 - 1) * 100:.2f}"


@pytest.mark.parametrize("bad_open", [0.0, float("nan")])
def test_a_bad_open_on_a_stop_day_fills_at_the_stop(cfg, bad_open):
    """Review of the fix: an Open of 0 gave a -100% exit, NaN a 'nan' return. A bad open isn't
    a price: the stop fills at the stop level."""
    h = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09", "2026-10-12"], [100, 100, 100, 100])
    h.loc["2026-10-12", ["Open", "Low"]] = (bad_open, 40)
    r = scanner.update_paper(pending_trade("X.ST", "2026-10-07"), {"X.ST": h}, h, cfg).iloc[0]
    stop = 100 * (1 + cfg["paper_trading"]["stop_loss"])
    assert (r["exit_reason"], r["exit_price"]) == ("stop", f"{stop:.4f}")
    assert r["pe_return_pct"] == f"{(stop / 100 - 1) * 100:.2f}"


def test_bad_open_on_a_target_day_fills_at_the_target(cfg):
    h = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09"], [100, 100, 100])
    h.loc["2026-10-09", ["Open", "High"]] = (0.0, 200)
    r = scanner.update_paper(pending_trade("X.ST", "2026-10-07"), {"X.ST": h}, h, cfg).iloc[0]
    target = 100 * (1 + cfg["paper_trading"]["take_profit"])
    assert (r["exit_reason"], r["exit_price"]) == ("target", f"{target:.4f}")


def test_a_missing_index_open_on_the_entry_day_is_a_gap():
    """Review of the fix: the gap check looked at the close only, so a stale value survived an
    index row with a close but no open on the entry day."""
    bench = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09"], [1000, 1010, 1030])
    bench.loc["2026-10-08", "Open"] = float("nan")
    rec = {c: "" for c in scanner.PAPER_COLUMNS}
    rec.update(
        id="Y",
        ticker="X.ST",
        status="closed",
        entry_date="2026-10-08",
        exit_date="2026-10-09",
        bench_entry="1000.0000",
        bench_exit="1030.0000",
        bench_return_pct="3.00",
        universe="v1",
    )
    p = scanner.refresh_benchmarks(pd.DataFrame([rec], columns=scanner.PAPER_COLUMNS), {"X.ST": STOCK}, bench)
    assert p.iloc[0]["bench_return_pct"] == ""


def test_partial_exit_index_figures_stay_with_their_trade():
    """Review of the fix: a closed variant row keeps its index figure (made with the fraction its
    return used); with the variant off, nothing is touched."""
    bench = _bars_on(["2026-10-07", "2026-10-08", "2026-10-09"], [1000, 1010, 1030])
    rec = {c: "" for c in scanner.PAPER_COLUMNS}
    rec.update(
        id="Y",
        ticker="X.ST",
        status="closed",
        entry_date="2026-10-08",
        exit_date="2026-10-09",
        pe_status="closed",
        pe_partial_date="2026-10-08",
        pe_exit_date="2026-10-09",
        pe_bench_return_pct="1.23",
        universe="v1",
    )
    df = pd.DataFrame([rec], columns=scanner.PAPER_COLUMNS)
    assert (
        scanner.refresh_benchmarks(df.copy(), {"X.ST": STOCK}, bench, None, 0.9).iloc[0]["pe_bench_return_pct"]
        == "1.23"
    )
    df.loc[0, "pe_status"], df.loc[0, "pe_bench_return_pct"] = "open", "1.23"
    assert (
        scanner.refresh_benchmarks(df.copy(), {"X.ST": STOCK}, bench, None, None).iloc[0]["pe_bench_return_pct"]
        == "1.23"
    )


def test_duplicate_index_dates_use_the_last_usable_value():
    b = pd.DataFrame(
        {"Open": [1000.0, float("nan")], "Close": [1001.0, 1002.0]},
        index=pd.DatetimeIndex(["2026-10-08", "2026-10-08"]),
    )
    assert scanner._price_exact(b, "2026-10-08", "Open") == 1000.0
    assert scanner._price_exact(b, "2026-10-08", "Close") == 1002.0
