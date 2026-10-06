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
               signal_date=signal_date, score="4", status="pending")
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
