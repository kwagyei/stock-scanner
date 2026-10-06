"""Offline tests for scanner.py. All fetchers are mocked; sockets are blocked."""
from __future__ import annotations

import datetime as dt
import socket

import pandas as pd
import pytest

import scanner

UTC = dt.timezone.utc
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
    monkeypatch.setattr(scanner, "DASHBOARD", tmp_path / "DASHBOARD.md")
    return tmp_path


def bars(rows, start="2026-06-01"):
    """rows: list of (open, high, low, close, volume) -> daily OHLCV on business days."""
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close", "Volume"])


def flat(n, px=100.0, vol=1000, start="2026-06-01"):
    return bars([(px, px, px, px, vol)] * n, start=start)


def news(source, hours_ago, title="t"):
    return {"title": title, "link": "http://x", "source": source,
            "ts": NOW - dt.timedelta(hours=hours_ago)}


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
    h.iloc[-1] = (100, 115, 100, 112, 6000)            # +12%, 6x volume
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
    h.iloc[-1, h.columns.get_loc("Close")] = 89     # -11% → +1
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


@pytest.mark.parametrize("pct,items,expected", [
    (0.50, [], True),
    (0.49, [], False),
    (0.80, [news("Cision News", 100)], False),     # release in window explains it
    (0.80, [news("Placera.se", 2)], True),         # media coverage is not a release
])
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
    assert news_calls == ["Good"]                  # no news requests for missing tickers


# --------------------------------------------------------------------------- paper trades

def pending_trade(ticker="X.ST", signal_date="2026-06-01"):
    rec = {c: "" for c in scanner.PAPER_COLUMNS}
    rec.update(id=f"{ticker}-{signal_date}", ticker=ticker, name="X",
               signal_date=signal_date, score="4", status="pending", universe="v1")
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
    full = bars([(100, 100, 100, 100, 1000),        # signal day
                 (102, 104, 101, 103, 1000),        # entry at open 102
                 (103, 105, 100, 104, 1000),
                 (104, 140, 103, 135, 1000)])       # target 132.6 hit intraday
    bench = bars([(1000, 1000, 1000, 1000, 0), (1010, 1010, 1010, 1010, 0),
                  (1010, 1010, 1010, 1020, 0), (1030, 1030, 1030, 1050, 0)])
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
    rows = [{"ticker": "A.ST", "name": "A", "date": "2026-06-02", "score": 6},
            {"ticker": "B.ST", "name": "B", "date": "2026-06-02", "score": 4},
            {"ticker": "C.ST", "name": "C", "date": "2026-06-02", "score": 3}]
    p = scanner.add_paper_signals(p, rows, cfg)
    assert list(p["ticker"]) == ["A.ST", "B.ST"]
    assert p.iloc[1]["status"] == "pending" and p.iloc[1]["id"] == "B.ST-2026-06-02"


# --------------------------------------------------------------------------- CSV round-trip

def test_paper_csv_round_trip(cfg, tmp_paths):
    scanner.DATA_DIR.mkdir()
    open_trade = scanner.update_paper(
        pending_trade("O.ST"), {"O.ST": flat(3)}, flat(3, px=1000), cfg)
    closed = run_trade(cfg, [(100, 101, 99, 100, 1), (80, 82, 78, 79, 1)]).to_frame().T
    df = pd.concat([open_trade, closed, pending_trade("P.ST")], ignore_index=True)

    df.to_csv(scanner.PAPER_LOG, index=False)
    back = scanner.load_paper()
    pd.testing.assert_frame_equal(back, df.astype(str), check_dtype=False)
    assert list(back["status"]) == ["open", "closed", "pending"]

    # Re-running the update on loaded string data is idempotent for closed trades.
    again = scanner.update_paper(back.copy(), {"X.ST": flat(3)}, flat(3), cfg)
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
    h.iloc[-1] = (100, 112, 100, 111, 6000)          # +11%, 6x vol, small cap → 4
    data = {"X.ST": h, "^OMX": flat(70, px=1000)}
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: data[t])
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: 1e9)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)

    scanner.main(["--no-email", "--watchlist", str(wl)])
    p = pd.read_csv(scanner.PAPER_LOG, dtype=str).fillna("")
    assert list(p[["ticker", "status", "score"]].iloc[0]) == ["X.ST", "pending", "4"]

    nxt = pd.bdate_range(h.index[-1] + pd.Timedelta(days=1), periods=2)
    data["X.ST"] = pd.concat([h, pd.DataFrame(
        [(111, 112, 110, 111, 1000), (90, 91, 80, 85, 1000)], index=nxt, columns=h.columns)])
    data["^OMX"] = flat(72, px=1000)
    scanner.main(["--no-email", "--watchlist", str(wl)])
    p = pd.read_csv(scanner.PAPER_LOG, dtype=str).fillna("")
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
    monkeypatch.setattr(scanner.feedparser, "parse",
                        lambda url, agent=None: urls.append(url) or real_parse(xml))
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
    return {"title": title, "link": "http://mfn", "source": "MFN",
            "ts": NOW - dt.timedelta(hours=hours_ago)}


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


def test_main_fetches_releases_only_with_slug(cfg, tmp_paths, monkeypatch):
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query,mfn_slug\nA.ST,A,A,a-co\nB.ST,B,B,\n")
    calls = []
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: flat(70))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_releases", lambda s, c: calls.append(s) or [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: None)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    scanner.main(["--no-email", "--watchlist", str(wl)])
    assert calls == ["a-co"]
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
    assert nm["n_24h"] == 1                        # still counted as news


# --------------------------------------------------------------------------- weekly summary

FRI_EVE = dt.datetime(2026, 10, 9, 18, 45)          # Friday, ISO week 41


@pytest.mark.parametrize("when,expected", [
    (FRI_EVE, True),
    (dt.datetime(2026, 10, 9, 8, 30), False),       # Friday morning run
    (dt.datetime(2026, 10, 8, 18, 45), False),      # Thursday
    (dt.datetime(2026, 10, 10, 12, 0), True),       # weekend manual run, week not yet done
])
def test_weekly_due_schedule(cfg, tmp_paths, when, expected):
    assert scanner.weekly_due(when, cfg) is expected


def test_weekly_due_once_per_week_unless_forced(cfg, tmp_paths):
    scanner.DATA_DIR.mkdir()
    scanner.WEEKLY_LOG.write_text("week,generated_local\n2026-W41,2026-10-06 20:23\n")
    assert scanner.weekly_due(FRI_EVE, cfg)            # early forced run doesn't count
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
        "2026-10-06T16:45,2026-10-06,B.ST,B,1,w\n")
    s = scanner.load_signals()
    assert list(zip(s["bar_date"], s["ticker"], s["score"])) == [
        ("2026-10-05", "A.ST", 3), ("2026-10-06", "A.ST", 4), ("2026-10-06", "B.ST", 1)]


def _signals(*rows):
    return pd.DataFrame([dict(zip(["bar_date", "ticker", "name", "score"], r)) for r in rows])


def test_signal_outcomes_vs_benchmark(cfg):
    h = bars([(100, 100, 100, c, 1) for c in (100, 110, 105, 120)])
    bench = bars([(1000, 1000, 1000, c, 0) for c in (1000, 1050, 1000, 1000)])
    sig = _signals(("2026-06-01", "X.ST", "X", 4), ("2026-06-03", "X.ST", "X", 3),
                   ("2026-06-01", "GONE.ST", "Gone", 5))
    o = scanner.signal_outcomes(sig, {"X.ST": h}, bench, [1, 2])
    assert o.loc[0, "ret_1d"] == pytest.approx(0.10) and o.loc[0, "excess_1d"] == pytest.approx(0.05)
    assert o.loc[0, "ret_2d"] == pytest.approx(0.05) and o.loc[0, "excess_2d"] == pytest.approx(0.05)
    assert o.loc[1, "ret_1d"] == pytest.approx(120 / 105 - 1)
    assert pd.isna(o.loc[1, "ret_2d"])                       # not enough bars yet
    assert o.loc[2, ["ret_1d", "ret_2d"]].isna().all()      # ticker without data


def test_hit_rate_table_buckets():
    o = pd.DataFrame({"score": [3, 3, 4, 6],
                      "ret_5d": [0.10, -0.05, 0.02, 0.30],
                      "excess_5d": [0.08, -0.06, -0.01, float("nan")]})
    t = {r["score"]: r for r in scanner.hit_rate_table(o, [5], top_score=5)}
    assert list(t) == ["all", "3", "4", "5+"]
    assert (t["all"]["signals"], t["all"]["n_5d"]) == (4, 3)
    assert t["all"]["hit_5d"] == pytest.approx(100 / 3)
    assert t["3"]["hit_5d"] == 50 and t["3"]["avg_5d"] == pytest.approx(2.5)
    assert t["5+"]["n_5d"] == 0 and t["5+"]["hit_5d"] is None and t["5+"]["avg_5d"] == pytest.approx(30)


def _trade(ticker, status, exit_date="", ret="", bench=""):
    rec = {c: "" for c in scanner.PAPER_COLUMNS}
    rec.update(ticker=ticker, name=ticker, status=status, exit_date=exit_date,
               return_pct=ret, bench_return_pct=bench)
    return rec


def test_weekly_paper_week_vs_all_time():
    paper = pd.DataFrame([
        _trade("A", "closed", "2026-09-20", "30.00", "2.00"),
        _trade("B", "closed", "2026-10-06", "-15.00", "1.00"),
        _trade("C", "closed", "2026-10-08", "10.00", "-1.00"),
        _trade("D", "open", ret="5.00"),
        _trade("E", "pending"),
    ], columns=scanner.PAPER_COLUMNS)
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
    start = (pd.Timestamp.today().normalize() - pd.offsets.BDay(69)).date()   # last bar = today
    h = flat(70, start=start)
    h.iloc[-1] = (100, 112, 100, 111, 6000)
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: h if t == "X.ST" else flat(70, 1000, start=start))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: 1e9)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    sent = []
    monkeypatch.setattr(scanner, "send_email", lambda html, subject: sent.append(subject))

    scanner.main(["--no-email", "--weekly", "--watchlist", str(wl)])
    scanner.main(["--no-email", "--weekly", "--watchlist", str(wl)])

    log = pd.read_csv(scanner.WEEKLY_LOG, dtype=str)
    assert len(log) == 1 and log.iloc[0]["signals_window"] == "1"
    html = (tmp_paths / "reports" / "weekly_latest.html").read_text()
    assert "Signal hit rate" in html and "X.ST" in html
    assert (tmp_paths / "reports" / f"weekly_{log.iloc[0]['week']}.html").exists()
    assert sent == []                                   # --no-email respected


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
    scanner.PAPER_LOG.write_text("id,ticker,status\nX-1,X.ST,pending\n")    # pre-versioning file
    assert scanner.load_paper().iloc[0]["universe"] == "v1"


def test_weekly_paper_split_by_universe():
    a = _trade("A", "closed", "2026-09-20", "30.00", "2.00")
    b = _trade("B", "closed", "2026-10-06", "-10.00", "1.00")
    a["universe"], b["universe"] = "v1", "auto-2026-10-06"
    wp = scanner.weekly_paper(pd.DataFrame([a, b], columns=scanner.PAPER_COLUMNS), dt.date(2026, 10, 5), 10000)
    assert wp["by_universe"]["v1"]["pnl_sek"] == pytest.approx(3000)
    assert wp["by_universe"]["auto-2026-10-06"]["avg_return"] == pytest.approx(-10)


# --------------------------------------------------------------------------- title filter

@pytest.mark.parametrize("word,title,keep", [
    ("Stille", "Pareto höjer riktkursen för Stille till 295 kronor, upprepar köp", True),
    ("Stille", "Stilles vd köper aktier", True),
    ("Stille", "Stille-aktien rusar", True),
    ("Stille", "Lördagsrysare i arenan: ”Jag gillar verkligen det vi gör”", False),
    ("Stille", "Ein stille Nacht", False),                       # case-sensitive
    ("I-Tech", "Strategen: Se möjligheterna utanför tech", False),
    ("I-Tech", "I-Tech tecknar nytt avtal för Selektope", True),
    ("K33", "K330 lanseras", False),
    ("Pricer", "Sifonen ersätter Quaero Capital i valberedningen i Pricer", True),
    ("Pricer", "Pricers vd: starkt kvartal", True),
])
def test_filter_titles(word, title, keep):
    assert (scanner.filter_titles([news("di.se", 1, title)], word) != []) is keep


def test_filter_titles_blank_keeps_all():
    items = [news("di.se", 1, "anything")]
    assert scanner.filter_titles(items, "") == items


def test_main_applies_title_match(cfg, tmp_paths, monkeypatch):
    wl = tmp_paths / "watchlist.csv"
    wl.write_text("ticker,name,query,title_match\nS.ST,Stille,Stille,Stille\nO.ST,Other,Other,\n")
    feed = [news("di.se", h, t) for h, t in ((1, "Stille rusar"), (2, "Rockbjörnens röda matta"), (3, "Seger för RIG Haga"))]
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: flat(70))
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: feed)
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: None)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    scanner.main(["--no-email", "--watchlist", str(wl)])
    log = pd.read_csv(scanner.SIGNALS_LOG).set_index("ticker")
    assert "S.ST" not in log.index                     # 1 relevant item: no spike, no signal
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
    assert r["pe_return_pct"] == "35.00"                     # 0.5 x 30% + 0.5 x 40%


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
    bench.iloc[1, 0] = 1000                                   # entry at the index open
    r = scanner.update_paper(pending_trade(), {"X.ST": h}, bench, cfg).iloc[0]
    assert r["pe_bench_return_pct"] == "15.00"               # 0.5 x 10% + 0.5 x 20%


@pytest.mark.parametrize("rows", [
    [D1, (110, 131, 109, 125, 1), (120, 121, 99, 100, 1)],
    [D1, (80, 82, 78, 79, 1)],
    [(100, 105, 95, 100 + i * 0.1, 1) for i in range(25)],
    [D1, (100, 110, 99, 108, 1)],
])
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

import evaluate as ev


def _row(**k):
    r = {"ticker": "A.ST", "name": "A | Co", "score": 6, "close": 10.0, "chg_1d": 0.12, "chg_5d": 0.2,
         "vol_ratio": 6.0, "n_24h": 2, "hype": False, "date": "2026-10-06",
         "reasons": ["company release <24h", "volume 6.0x avg", "price +12% 1d", "small cap"],
         "points": {"release": 2, "volume": 2, "price": 1, "news": 0, "small_cap": 1},
         "market_cap": 1.5e9, "n_releases": 1,
         "headlines": [{"title": "A | Co wins order", "link": "https://x.test/a?b=(1)"}]}
    r.update(k)
    return r


@pytest.mark.parametrize("raw,nice", [
    ("company release <24h", "📣 Company release"),
    ("volume 5.1x avg", "📊 Volume 5.1× normal"),
    ("price -11% 1d", "📉 Big drop ▼ 11%"),
    ("price +10% 1d", "📈 Big jump ▲ 10%"),
    ("news spike (3 in 24h vs 1.0/day)", "📰 3 news items today (usually 1.0/day)"),
    ("small cap", "🔹 Small cap"),
    ("something new", "something new"),
])
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


@pytest.mark.parametrize("pct,out", [(2.5, "▲ 2.5%"), (-11.2, "▼ 11.2%"), (0.01, "0.0%"), (None, "–"),
                                     (float("nan"), "–")])
def test_arrow(pct, out):
    assert scanner._arrow(pct) == out


def test_alert_table_shows_component_values():
    t = scanner._alerts_table([_row()], "")
    head, sep, row = t.splitlines()[:3]
    assert "📣<br><sub>Releases 24h</sub>" in head and "🔹<br><sub>Market cap</sub>" in head
    assert "Today" not in head                                   # price column replaces it
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
    assert "**A \\| Co**<br><sub>A.ST</sub>" in t and "▲ 12.0%" in t and "📣 Company release" in t
    h = scanner._headlines([_row()])
    assert h.startswith("<details>") and "](<https://x.test/a?b=(1)>)" in h and "A \\| Co wins order" in h


def test_nav_buttons(monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    root, sub = scanner._nav("DASHBOARD.md"), scanner._nav("reports/latest.md")
    assert "badge/Dashboard-d0d7de" in root and "(reports/latest.md)" in root      # current page greyed
    assert "(../DASHBOARD.md)" in sub and "(latest.md)" not in sub and "(../reports/latest.md)" in sub
    assert "badge/Paper%20trades-" in root and "Runs" not in root
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
    monkeypatch.setenv("GITHUB_REPOSITORY", "me/repo")
    assert "(https://github.com/me/repo/actions)" in scanner._nav("DASHBOARD.md")


def _paper_fixture():
    o = _trade("O.ST", "open", ret="4.00", bench="1.00")
    o.update(entry_date="2026-10-01", entry_price="100.0000", last_price="104.0000", pe_status="open")
    c = _trade("C.ST", "closed", "2026-10-02", "30.00", "1.00")
    c.update(entry_date="2026-09-20", exit_reason="target", pe_status="closed", pe_return_pct="15.00",
             pe_exit_reason="remainder stop")
    p = _trade("P.ST", "pending")
    p.update(signal_date="2026-10-06", score="5")
    return pd.DataFrame([o, c, p], columns=scanner.PAPER_COLUMNS)


def test_dashboard(cfg, tmp_paths, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    scanner.DATA_DIR.mkdir(); scanner.REPORT_DIR.mkdir()
    scanner.SIGNALS_LOG.write_text("run_utc,bar_date,ticker,name,score,reasons\n"
                                   "2026-10-06T16:45,2026-10-06,A.ST,A,4,volume 6.0x avg\n"
                                   "2026-10-06T16:45,2026-10-06,B.ST,B,1,small cap\n")
    scanner.WEEKLY_LOG.write_text("week,signals_week,hit_5d,n_5d,hit_20d,n_20d,paper_all_n,paper_all_pnl_sek\n"
                                  "2026-W41,3,50.0,2,,0,1,1500\n")
    (scanner.REPORT_DIR / "weekly_2026-W41.md").write_text("x")
    res = ev.evaluate(ev.closed_trades(_paper_fixture()))
    md = scanner.build_dashboard([_row(), _row(ticker="H.ST", score=0, hype=True)], ["BAD.ST"],
                                 _paper_fixture(), cfg, dt.datetime(2026, 10, 7, 18, 45), res)
    for s in ["# 📊 Attention scanner", "> [!WARNING]\n> No price data for: BAD.ST",
              "| 🔔 Alerts | ⚠️ Hype warnings | 💼 Open trades | ✅ Closed trades | ⚖️ Evaluation |",
              "| **1**<br><sub>score ≥ 3</sub> | **1** | **1**<br><sub>+1 waiting</sub> | **1** | **NOT YET**",
              "## ⚠️ Hype warnings", "> [!CAUTION]",
              "🎯 130.00<br>🛑 85.00", "| 5 of 20 |", "100.00 → 104.00", "⏳ buys at the next open",
              "> [!NOTE]\n> **Evaluation: NOT YET – 1 of 30 trades**",
              "[2026-W41](reports/weekly_2026-W41.md)", "50% <sub>of 2</sub>", "+1 500 SEK",
              "(reports/paper_trades.md)", "(reports/evaluation.md)", "(data/signals_log.csv)"]:
        assert s in md, s
    recent = md.split("Signals in the last 10 trading days")[1].split("</details>")[0]
    assert "A.ST" in recent and "B.ST" not in recent              # score 1 isn't a signal
    assert md.count("<details>") == md.count("</details>")


def test_dashboard_empty(cfg, tmp_paths, monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    md = scanner.build_dashboard([], [], pd.DataFrame(columns=scanner.PAPER_COLUMNS), cfg,
                                 dt.datetime(2026, 10, 6, 8, 30), ev.evaluate(None))
    for s in ["✅ No alerts on this run.", "No signals yet.", "first one comes on Friday", "No open positions."]:
        assert s in md, s
    assert "## ⚠️ Hype warnings" not in md and "Waiting to buy" not in md


def test_paper_page(cfg):
    paper = _paper_fixture()
    md = scanner.build_paper_md(paper, cfg, dt.datetime(2026, 10, 7, 18, 45), ev.evaluate(ev.closed_trades(paper)))
    for s in ["# 💼 Paper trades", "| **Main rules** | 2 | 1 | 100% | ▲ 30.0% | ▲ 29.0% | +3 000 SEK |",
              "| **Partial-exit variant** | 1 | 1 |", "## Open (1)", "## Waiting to buy (1)", "## Closed (1)",
              "2026-09-20 → 2026-10-02", "🎯 Target", "▲ 15.0%<br><sub>🛑 Breakeven stop</sub>"]:
        assert s in md, s


def _eval_res(tmp_path, excess, n):
    rows = [{"ticker": f"T{i}", "status": "closed", "return_pct": excess[i % len(excess)] + 1,
             "bench_return_pct": 0, "exit_reason": "time", "exit_date": f"2026-01-{i % 28 + 1:02d}"}
            for i in range(n)]
    path = tmp_path / "p.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return ev.evaluate(ev.load_trades(path))


def test_evaluation_page_verdicts(cfg, tmp_path):
    now = dt.datetime(2026, 10, 7, 18, 45)
    passed = scanner.build_evaluation_md(_eval_res(tmp_path, [1, 5], 40), None, cfg, now)
    assert "> [!TIP]\n> **Evaluation: PASS – 40 trades" in passed and "| ✅ | Enough trades | 40 closed (need 30) |" in passed
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
    titles = {scanner.DASHBOARD: "# 📊", rep / "latest.md": "# 📰 Today", rep / "paper_trades.md": "# 💼",
              rep / "evaluation.md": "# ⚖️ Evaluation", rep / "weekly_latest.md": "# 📅 Weekly summary"}
    for path, title in titles.items():
        text = path.read_text()
        assert text.startswith(title), path
        assert "img.shields.io/badge/Dashboard" in text, path               # button row on every page
        assert text.count("<details>") == text.count("</details>"), path
    assert len(list(rep.glob("weekly_20*.md"))) == 1


def test_weekly_with_no_signals_in_window(cfg, tmp_paths):
    scanner.DATA_DIR.mkdir(); scanner.REPORT_DIR.mkdir()
    html, md = scanner.run_weekly(pd.DataFrame(columns=scanner.PAPER_COLUMNS), {}, flat(5), cfg, FRI_EVE)
    assert "No signals in the window yet." in md and "No signals this week." in md
    assert "img.shields.io" not in md                    # job-summary version has no button row
    assert "img.shields.io" in (scanner.REPORT_DIR / "weekly_latest.md").read_text()


# --------------------------------------------------------------------------- score components / score log

def _scenario(vol=1000, close=100, items=(), mcap=None):
    h = flat(70)
    h.iloc[-1] = (100, max(100, close), min(100, close), close, vol)
    return (scanner.price_metrics(h, CFG_FOR_SCENARIOS), scanner.news_metrics(list(items), NOW, CFG_FOR_SCENARIOS), mcap)


CFG_FOR_SCENARIOS = scanner.load_config(scanner.ROOT / "config.yaml")


@pytest.mark.parametrize("kw,expected", [
    ({}, {}),
    ({"vol": 3500}, {"volume": 1}),
    ({"vol": 6000, "close": 112, "mcap": 1e9}, {"volume": 2, "price": 1, "small_cap": 1}),
    ({"mcap": 1e9}, {}),                                                   # small cap alone: nothing
    ({"items": [news("Cision News", 2), news("di.se", 3), news("EFN", 5)], "mcap": 1e9},
     {"release": 2, "news": 1, "small_cap": 1}),
    ({"close": 85, "mcap": 9e9}, {"price": 1}),                            # big cap: no small-cap point
])
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
    assert (h["score"], h["pts_volume"], h["pts_price"], h["pts_small_cap"], h["pts_release"], h["pts_news"]) == \
        ("4", "2", "1", "1", "0", "0")
    assert (h["status"], h["market_cap_msek"], h["universe"], h["news_24h"]) == ("ok", "1500", "auto-test", "0")
    q = log.loc["QUIET.ST"]
    assert (q["score"], q["pts_small_cap"], q["reasons"]) == ("0", "0", "")    # logged although quiet
    g = log.loc["GONE.ST"]
    assert (g["status"], g["score"], g["name"]) == ("no price data", "", "Gone")
    sig = pd.read_csv(scanner.SIGNALS_LOG)
    assert list(sig["ticker"]) == ["HOT.ST"]                                     # signals log unchanged

    scanner.main(["--no-email", "--no-news", "--watchlist", str(wl)])
    log = pd.read_csv(scanner.SCORES_LOG, dtype=str).fillna("")
    assert len(log) == 6 and log["run_utc"].nunique() >= 1
    assert log.iloc[3]["news_24h"] == "" and log.iloc[3]["releases_24h"] == ""   # news not checked


def test_append_csv_rewrites_when_columns_change(tmp_path):
    path = tmp_path / "log.csv"
    scanner.append_csv(path, [{"a": 1, "b": 2}])
    scanner.append_csv(path, [{"a": 3, "b": 4}])
    scanner.append_csv(path, [{"a": 5, "b": 6, "c": 7}])
    df = pd.read_csv(path, dtype=str).fillna("")
    assert list(df.columns) == ["a", "b", "c"] and df["c"].tolist() == ["", "", "7"]
    scanner.append_csv(path, [])
    assert len(pd.read_csv(path)) == 3
