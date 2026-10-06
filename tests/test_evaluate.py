"""Tests for evaluate.py with synthetic paper-trade logs. No network."""
import datetime as dt

import pandas as pd
import pytest

import evaluate as ev
import scanner


def log(tmp_path, trades, name="paper_trades.csv"):
    """trades: list of (return_pct, bench_return_pct, exit_reason, exit_date[, status])."""
    rows = []
    for i, t in enumerate(trades):
        ret, bench, reason, date, *status = t
        rows.append({"id": f"T{i}", "ticker": f"T{i}.ST", "status": status[0] if status else "closed",
                     "return_pct": "" if ret is None else f"{ret}", "bench_return_pct": bench,
                     "exit_reason": reason, "exit_date": date})
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def dates(n):
    return [(dt.date(2026, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(n)]


# Four trades, hand-computed. cost 1 -> net 10, 5, -5, 0; bench 1, 0 (missing), 0, 1
# -> excess 9, 5, -5, -1 (exit-date order; the file is deliberately shuffled).
HAND = [(-4, "0", "stop", "2026-01-03"), (11, "1", "target", "2026-01-01"),
        (1, "1", "time", "2026-01-04"), (6, "", "time", "2026-01-02")]


def test_hand_computed_statistics(tmp_path):
    r = ev.evaluate(ev.load_trades(log(tmp_path, HAND)), cost=1.0, stake=0.10, min_trades=4)
    assert r["n"] == 4
    assert r["avg_net"] == pytest.approx(2.5) and r["avg_bench"] == pytest.approx(0.5)
    assert r["avg_excess"] == pytest.approx(2.0) and r["median_excess"] == pytest.approx(2.0)
    assert r["excess_wo_best"] == pytest.approx(-1 / 3) and r["excess_wo_best2"] == pytest.approx(-3.0)
    assert r["win_rate"] == 50 and r["avg_win"] == pytest.approx(7.5) and r["avg_loss"] == pytest.approx(-2.5)
    assert r["losing_streak"] == 2                       # -5 then 0, in exit-date order
    assert r["max_drawdown"] == pytest.approx(-0.5)     # 1.01505 -> 1.00997475
    assert r["t_stat"] == pytest.approx(2.0 / ((116 / 3) ** 0.5 / 2))
    time = r["by_reason"].loc["time"]
    assert (time["trades"], time["avg_net"], time["avg_excess"]) == (2, 2.5, 2.0)
    assert r["verdict"] == "FAIL"                        # excess gone without the best trade
    assert not r["checks"]["still beats it without the best trade"]


def test_cost_and_stake_flags(tmp_path):
    t = ev.load_trades(log(tmp_path, HAND))
    a, b = ev.evaluate(t, cost=0.0), ev.evaluate(t, cost=2.5)
    assert a["avg_net"] - b["avg_net"] == pytest.approx(2.5)
    assert ev.evaluate(t, stake=0.2)["max_drawdown"] < ev.evaluate(t, stake=0.1)["max_drawdown"]


def test_pass(tmp_path):
    trades = [(1 + 1 + 3 + (2 if i % 2 else -2), "1", "time", d) for i, d in enumerate(dates(40))]
    r = ev.evaluate(ev.load_trades(log(tmp_path, trades)))
    assert r["verdict"] == "PASS" and all(r["checks"].values())
    assert r["avg_excess"] == pytest.approx(3.0) and r["t_stat"] > 2 and r["p_positive"] == 1.0
    assert ev.one_line(r).startswith("Evaluation: PASS – 40 trades, +3.0% vs index")


def test_fail_only_on_t_statistic(tmp_path):
    # excess alternates +6 / -4.5: positive on average and without the best trade, but noisy
    trades = [((7 if i % 2 else -3.5), "0", "time", d) for i, d in enumerate(dates(40))]
    r = ev.evaluate(ev.load_trades(log(tmp_path, trades)))
    assert r["avg_excess"] == pytest.approx(0.75) and r["excess_wo_best"] > 0
    assert 0 < r["t_stat"] < 2
    assert [k for k, ok in r["checks"].items() if not ok] == ["unlikely to be luck (t > 2)"]
    assert r["verdict"] == "FAIL"


def test_fail_one_lucky_trade(tmp_path):
    trades = [(101, "0", "target", "2026-03-01")] + [(0, "0", "stop", d) for d in dates(29)]
    r = ev.evaluate(ev.load_trades(log(tmp_path, trades)))
    assert r["avg_excess"] > 0 and r["excess_wo_best"] == pytest.approx(-1.0)
    assert r["verdict"] == "FAIL"


def test_not_yet(tmp_path):
    trades = [(5, "1", "time", d) for d in dates(12)]
    r = ev.evaluate(ev.load_trades(log(tmp_path, trades)))
    assert r["verdict"] == "NOT YET" and ev.one_line(r) == "Evaluation: NOT YET – 12 of 30 trades"
    assert "12 of 30 trades so far" in ev.format_report(r)


def test_no_file(tmp_path):
    r = ev.evaluate(ev.load_trades(tmp_path / "missing.csv"))
    assert r["verdict"] == "NOT YET" and r["file_missing"]
    assert "No paper-trade log found" in ev.format_report(r)
    assert ev.one_line(r) == "Evaluation: NOT YET – 0 of 30 trades"


def test_zero_closed(tmp_path):
    path = log(tmp_path, [(None, "", "", "", "pending"), (3, "", "", "", "open")])
    r = ev.evaluate(ev.load_trades(path))
    assert (r["n"], r["verdict"]) == (0, "NOT YET")
    assert "No closed paper trades yet" in ev.format_report(r)


def test_one_closed_trade(tmp_path):
    r = ev.evaluate(ev.load_trades(log(tmp_path, [(20, "", "target", "2026-01-01")])), min_trades=1)
    assert r["t_stat"] is None and r["p_positive"] is None and r["excess_wo_best"] is None
    assert r["verdict"] == "FAIL"                     # can't show it isn't luck with one trade
    report = ev.format_report(r)
    assert "t-statistic of the excess          –" in report and "+19.0%" in report


def test_missing_bench_column_counts_as_zero(tmp_path):
    path = tmp_path / "p.csv"
    path.write_text("ticker,status,return_pct,exit_reason,exit_date\nA,closed,5,time,2026-01-01\n")
    r = ev.evaluate(ev.load_trades(path), min_trades=1)
    assert r["avg_bench"] == 0 and r["avg_excess"] == pytest.approx(4.0)


def test_bootstrap_is_reproducible(tmp_path):
    t = ev.load_trades(log(tmp_path, HAND))
    assert ev.evaluate(t)["p_positive"] == ev.evaluate(t)["p_positive"]
    assert 0 < ev.evaluate(t)["p_positive"] < 1


def test_variant_uses_pe_columns():
    paper = pd.DataFrame([{"ticker": "A", "status": "closed", "return_pct": "30", "bench_return_pct": "0",
                           "exit_reason": "target", "exit_date": "2026-01-02",
                           "pe_status": "closed", "pe_return_pct": "15", "pe_bench_return_pct": "1",
                           "pe_exit_reason": "remainder stop", "pe_exit_date": "2026-01-05"}])
    main, var = ev.closed_trades(paper), ev.closed_trades(paper, variant=True)
    assert (main["ret"][0], var["ret"][0], var["reason"][0]) == (30, 15, "remainder stop")


def test_cli(tmp_path, capsys):
    path = log(tmp_path, HAND)
    assert ev.main(["--file", str(path), "--min-trades", "4", "--cost", "1"]) == 0
    out = capsys.readouterr().out
    assert "VERDICT: FAIL" in out and "✗ still beats it without the best trade: -0.3% per trade" in out
    assert "✓ enough trades: 4 (need 4)" in out
    assert "By exit reason" in out and "Longest losing streak              2 trade(s)" in out


def test_config_defaults_match_spec():
    assert ev.defaults_from_config()["cost_pct"] == 1.0
    assert ev.defaults_from_config()["stake"] == 0.10
    assert ev.defaults_from_config()["min_trades"] == 30


# --------------------------------------------------------------------------- digest integration

@pytest.fixture
def scan_env(tmp_path, monkeypatch):
    for name, sub in [("DATA_DIR", "data"), ("REPORT_DIR", "reports")]:
        monkeypatch.setattr(scanner, name, tmp_path / sub)
    for name, f in [("SIGNALS_LOG", "signals_log.csv"), ("PAPER_LOG", "paper_trades.csv"),
                    ("WEEKLY_LOG", "weekly_summary.csv"), ("SCORES_LOG", "scores_log.csv")]:
        monkeypatch.setattr(scanner, name, tmp_path / "data" / f)
    monkeypatch.setattr(scanner, "DASHBOARD", tmp_path / "DASHBOARD.md")
    flat = pd.DataFrame([(10, 10, 10, 10, 1000)] * 70, index=pd.bdate_range("2026-06-01", periods=70),
                        columns=["Open", "High", "Low", "Close", "Volume"])
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: flat)
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: None)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    summary = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    wl = tmp_path / "watchlist.csv"
    wl.write_text("ticker,name,query\nX.ST,X,X\n")
    return tmp_path, wl, summary


@pytest.mark.parametrize("utc_hour,in_digest", [(16, True), (6, False)])   # 18:45 / 08:30 Stockholm
def test_verdict_line_in_evening_digest(scan_env, monkeypatch, utc_hour, in_digest):
    tmp, wl, summary = scan_env
    monkeypatch.setattr(scanner, "_now", lambda: dt.datetime(2026, 10, 7, utc_hour, 45, tzinfo=dt.timezone.utc))
    scanner.main(["--no-email", "--watchlist", str(wl)])
    line = "Evaluation: NOT YET – 0 of 30 trades"
    assert (line in (tmp / "reports" / "latest.html").read_text()) is in_digest
    assert (line in summary.read_text()) is in_digest
    assert line in (tmp / "DASHBOARD.md").read_text()            # dashboard always shows it
