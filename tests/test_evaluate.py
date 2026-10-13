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
        rows.append(
            {
                "id": f"T{i}",
                "ticker": f"T{i}.ST",
                "status": status[0] if status else "closed",
                "return_pct": "" if ret is None else f"{ret}",
                "bench_return_pct": bench,
                "exit_reason": reason,
                "exit_date": date,
            }
        )
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def dates(n):
    return [(dt.date(2026, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(n)]


# Four trades, hand-computed. cost 1 -> net 10, 5, -5, 0; bench 1, 0, 0, 1
# -> excess 9, 5, -5, -1 (exit-date order; the file is deliberately shuffled).
HAND = [
    (-4, "0", "stop", "2026-01-03"),
    (11, "1", "target", "2026-01-01"),
    (1, "1", "time", "2026-01-04"),
    (6, "0", "time", "2026-01-02"),
]


def test_hand_computed_statistics(tmp_path):
    r = ev.evaluate(ev.load_trades(log(tmp_path, HAND)), cost=1.0, stake=0.10, min_trades=4)
    assert r["n"] == 4
    assert r["avg_net"] == pytest.approx(2.5) and r["avg_bench"] == pytest.approx(0.5)
    assert r["avg_excess"] == pytest.approx(2.0) and r["median_excess"] == pytest.approx(2.0)
    assert r["excess_wo_best"] == pytest.approx(-1 / 3) and r["excess_wo_best2"] == pytest.approx(-3.0)
    assert r["win_rate"] == 50 and r["avg_win"] == pytest.approx(7.5) and r["avg_loss"] == pytest.approx(-2.5)
    assert r["losing_streak"] == 2  # -5 then 0, in exit-date order
    assert r["max_drawdown"] == pytest.approx(-0.5)  # 1.01505 -> 1.00997475
    assert r["t_stat"] == pytest.approx(2.0 / ((116 / 3) ** 0.5 / 2))
    time = r["by_reason"].loc["time"]
    assert (time["trades"], time["avg_net"], time["avg_excess"]) == (2, 2.5, 2.0)
    assert r["verdict"] == "FAIL"  # excess gone without the best trade
    assert not r["checks"]["still beats it without the best trade"]


def test_cost_and_stake_flags(tmp_path):
    t = ev.load_trades(log(tmp_path, HAND))
    a, b = ev.evaluate(t, cost=0.0), ev.evaluate(t, cost=2.5)
    assert a["avg_net"] - b["avg_net"] == pytest.approx(2.5)
    assert ev.evaluate(t, stake=0.2)["max_drawdown"] < ev.evaluate(t, stake=0.1)["max_drawdown"]


def test_pass(tmp_path):
    trades = [(1 + 1 + 3 + (2 if i % 2 else -2), "1", "time", d) for i, d in enumerate(dates(40))]
    r = ev.evaluate(ev.load_trades(log(tmp_path, trades)), cost=1.0)
    assert r["verdict"] == "PASS" and all(r["checks"].values())
    assert r["avg_excess"] == pytest.approx(3.0) and r["t_stat"] > 2 and r["p_positive"] == 1.0
    assert ev.one_line(r).startswith("Evaluation: PASS – 40 trades, +3.0% vs index")


def test_fail_only_on_t_statistic(tmp_path):
    # excess alternates +6 / -4.5: positive on average and without the best trade, but noisy
    trades = [((7 if i % 2 else -3.5), "0", "time", d) for i, d in enumerate(dates(40))]
    r = ev.evaluate(ev.load_trades(log(tmp_path, trades)), cost=1.0)
    assert r["avg_excess"] == pytest.approx(0.75) and r["excess_wo_best"] > 0
    assert 0 < r["t_stat"] < 2
    assert [k for k, ok in r["checks"].items() if not ok] == ["unlikely to be luck (t > 2)"]
    assert r["verdict"] == "FAIL"


def test_fail_one_lucky_trade(tmp_path):
    trades = [(101, "0", "target", "2026-03-01")] + [(0, "0", "stop", d) for d in dates(29)]
    r = ev.evaluate(ev.load_trades(log(tmp_path, trades)), cost=1.0)
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
    r = ev.evaluate(ev.load_trades(log(tmp_path, [(20, "0", "target", "2026-01-01")])), min_trades=1, cost=1.0)
    assert r["t_stat"] is None and r["p_positive"] is None and r["excess_wo_best"] is None
    assert r["verdict"] == "FAIL"  # can't show it isn't luck with one trade
    report = ev.format_report(r)
    assert "t-statistic of the excess          –" in report and "+19.0%" in report


def test_trades_without_any_benchmark_are_left_out_and_counted(tmp_path):
    """No index return over a trade's days -> it can't be judged against the index. Counting it
    as 0% would quietly favour or penalise it, so it's left out and reported."""
    path = tmp_path / "p.csv"
    path.write_text(
        "ticker,status,return_pct,bench_return_pct,bench_primary_return_pct,exit_reason,exit_date\n"
        "A,closed,5,,,time,2026-01-01\n"  # neither benchmark
        "B,closed,8,2,,time,2026-01-02\n"  # secondary only
        "C,closed,3,1,-1,time,2026-01-03\n"  # both: primary is used
    )
    r = ev.evaluate(ev.load_trades(path), min_trades=1, cost=1.0)
    assert r["n"] == 2 and r["n_no_bench"] == 1
    assert r["avg_excess"] == pytest.approx(((8 - 1 - 2) + (3 - 1 + 1)) / 2)
    assert r["avg_excess_secondary"] == pytest.approx(((8 - 1 - 2) + (3 - 1 - 1)) / 2)
    assert "1 closed trade(s) left out" in ev.format_markdown(r)
    nothing = tmp_path / "q.csv"
    nothing.write_text("ticker,status,return_pct,exit_reason,exit_date\nA,closed,5,time,2026-01-01\n")
    r = ev.evaluate(ev.load_trades(nothing), min_trades=1, cost=1.0)
    assert r["n"] == 0 and r["n_no_bench"] == 1 and r["verdict"] == "NOT YET"


def test_bootstrap_is_reproducible(tmp_path):
    t = ev.load_trades(log(tmp_path, HAND))
    assert ev.evaluate(t)["p_positive"] == ev.evaluate(t)["p_positive"]
    assert 0 < ev.evaluate(t)["p_positive"] < 1


def test_variant_uses_pe_columns():
    paper = pd.DataFrame(
        [
            {
                "ticker": "A",
                "status": "closed",
                "return_pct": "30",
                "bench_return_pct": "0",
                "exit_reason": "target",
                "exit_date": "2026-01-02",
                "pe_status": "closed",
                "pe_return_pct": "15",
                "pe_bench_return_pct": "1",
                "pe_exit_reason": "remainder stop",
                "pe_exit_date": "2026-01-05",
            }
        ]
    )
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
    assert ev.defaults_from_config()["cost_pct"] == pytest.approx(1.2)  # 2 x (0.1 + 0.5), costs.* in config
    assert ev.defaults_from_config()["stake"] == 0.10
    assert ev.defaults_from_config()["min_trades"] == 30


# --------------------------------------------------------------------------- digest integration


@pytest.fixture
def scan_env(tmp_path, monkeypatch):
    for name, sub in [("DATA_DIR", "data"), ("REPORT_DIR", "reports")]:
        monkeypatch.setattr(scanner, name, tmp_path / sub)
    for name, f in [
        ("SIGNALS_LOG", "signals_log.csv"),
        ("PAPER_LOG", "paper_trades.csv"),
        ("WEEKLY_LOG", "weekly_summary.csv"),
        ("SCORES_LOG", "scores_log.csv"),
    ]:
        monkeypatch.setattr(scanner, name, tmp_path / "data" / f)
    monkeypatch.setattr(scanner, "DASHBOARD", tmp_path / "DASHBOARD.md")
    monkeypatch.setattr(scanner, "FUND_SUMMARY", tmp_path / "fund_watch_latest.json")
    flat = pd.DataFrame(
        [(10, 10, 10, 10, 1000)] * 70,
        index=pd.bdate_range("2026-06-01", periods=70),
        columns=["Open", "High", "Low", "Close", "Volume"],
    )
    monkeypatch.setattr(scanner, "fetch_history", lambda t, p: flat)
    monkeypatch.setattr(scanner, "fetch_news", lambda q, c: [])
    monkeypatch.setattr(scanner, "fetch_market_cap", lambda t: None)
    monkeypatch.setattr(scanner.time, "sleep", lambda s: None)
    summary = tmp_path / "step_summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    wl = tmp_path / "watchlist.csv"
    wl.write_text("ticker,name,query\nX.ST,X,X\n")
    return tmp_path, wl, summary


@pytest.mark.parametrize("utc_hour,in_digest", [(16, True), (6, False)])  # 18:45 / 08:30 Stockholm
def test_verdict_line_in_evening_digest(scan_env, monkeypatch, utc_hour, in_digest):
    tmp, wl, summary = scan_env
    monkeypatch.setattr(scanner, "_now", lambda: dt.datetime(2026, 10, 7, utc_hour, 45, tzinfo=dt.UTC))
    scanner.main(["--no-email", "--watchlist", str(wl)])
    line = "Evaluation: NOT YET – 0 of 30 trades"
    assert (line in (tmp / "reports" / "latest.html").read_text()) is in_digest
    assert (line in summary.read_text()) is in_digest
    assert line in (tmp / "DASHBOARD.md").read_text()  # dashboard always shows it


# --------------------------------------------------------------------------- direction split


@pytest.mark.parametrize(
    "move,d",
    [
        (-0.112, "drop"),
        (-0.10, "drop"),
        (-0.0999, "flat"),
        (0.0, "flat"),
        (0.10, "jump"),
        (0.25, "jump"),
        ("", "unknown"),
        (None, "unknown"),
        (float("nan"), "unknown"),
        ("-0.15", "drop"),
    ],
)
def test_direction(move, d):
    assert ev.direction(move, 0.10) == d


def _moves_log(tmp_path):
    # (return, signal move): two drops (one win), one jump (win), one flat (loss), one unknown
    rows = [(11, "-0.12"), (-9, "-0.15"), (21, "0.11"), (-4, "0.03"), (6, "")]
    df = pd.DataFrame(
        [
            {
                "ticker": f"T{i}",
                "status": "closed",
                "return_pct": r,
                "bench_return_pct": 0,
                "exit_reason": "time",
                "exit_date": f"2026-01-0{i + 1}",
                "signal_move": m,
            }
            for i, (r, m) in enumerate(rows)
        ]
    )
    path = tmp_path / "p.csv"
    df.to_csv(path, index=False)
    return path


def test_by_direction_breakdown(tmp_path):
    r = ev.evaluate(ev.load_trades(_moves_log(tmp_path)), cost=1.0)
    bd = r["by_direction"]
    assert list(bd.index) == ["drop", "jump", "flat", "unknown"]
    assert (bd.loc["drop", "trades"], bd.loc["drop", "win_rate"], bd.loc["drop", "avg_net"]) == (2, 50.0, 0.0)
    assert (bd.loc["jump", "avg_net"], bd.loc["flat", "win_rate"]) == (20.0, 0.0)
    assert "By signal-day move" in ev.format_report(r) and "▼ Drop" in ev.format_report(r)
    md = ev.format_markdown(r)
    assert "### By signal-day move" in md and "| ▲ Jump | 1 | 100% | +20.0% | +20.0% |" in md


def test_cli_direction_filter(tmp_path, capsys):
    ev.main(["--file", str(_moves_log(tmp_path)), "--direction", "drop", "--min-trades", "1"])
    out = capsys.readouterr().out
    assert "signal day: ▼ Drop" in out and "Per trade, after costs (2 closed trades)" in out


# --------------------------------------------------------------------------- costs, benchmarks, strategies


def test_trade_costs_use_logged_spread_else_config():
    c = ev.trade_costs(pd.Series(["1.0", "", "0.4"]), {"commission_pct": 0.1, "half_spread_pct": 0.5})
    assert list(c) == pytest.approx([1.2, 1.2, 0.6])
    assert ev.round_trip({"commission_pct": 0.1, "half_spread_pct": 0.5}) == pytest.approx(1.2)


def _books():
    rows = []
    for strat, rets in (("", [10, 4]), ("v0b", [6, -2]), ("v1_v1", [8, 8])):
        for i, r in enumerate(rets):
            rows.append(
                {
                    "ticker": f"T{i}",
                    "status": "closed",
                    "return_pct": str(r),
                    "bench_return_pct": "1",
                    "bench_primary_return_pct": "2" if i == 0 else "",
                    "exit_reason": "time",
                    "exit_date": f"2026-11-0{i + 1}",
                    "signal_move": "0",
                    "strategy": strat,
                    "spread_pct": "0.8" if i == 0 else "",
                }
            )
    return pd.DataFrame(rows)


def test_closed_trades_per_strategy_primary_benchmark_and_costs():
    t = ev.closed_trades(_books(), strategy="v0", costs={"commission_pct": 0.1, "half_spread_pct": 0.5})
    assert list(t["ret"]) == [10, 4]  # blank strategy = v0
    assert list(t["bench"]) == [2, 1]  # primary, else secondary
    assert list(t["bench_secondary"]) == [1, 1]
    assert list(t["cost"]) == pytest.approx([1.0, 1.2])  # 0.2 + spread 0.8 / 0.2 + 1.0
    r = ev.evaluate(t, min_trades=2)
    assert r["cost"] == pytest.approx(1.1)
    assert r["avg_excess"] == pytest.approx(((10 - 1.0 - 2) + (4 - 1.2 - 1)) / 2)
    assert r["avg_excess_secondary"] == pytest.approx(((10 - 1.0 - 1) + (4 - 1.2 - 1)) / 2)


def test_compare_gives_one_verdict_per_strategy():
    d = {**ev.defaults_from_config(), "min_trades": 2}
    cmp = ev.compare(_books(), None, ["v0", "v0b", "v1_v1"], d)
    assert {s: r["n"] for s, r in cmp["strategies"].items()} == {"v0": 2, "v0b": 2, "v1_v1": 2}
    assert cmp["strategies"]["v1_v1"]["avg_net"] > cmp["strategies"]["v0b"]["avg_net"]
    md = ev.format_compare_markdown(cmp)
    assert "| `v0` | `v0b` | `v1_v1` |" in md and "Avg vs small caps" in md and "No scored releases yet" in md
    few = ev.compare(_books(), None, ["v0", "v0b", "v1_v1"], {**d, "min_trades": 30})
    assert {r["verdict"] for r in few["strategies"].values()} == {"NOT YET"}


def test_strategy_labels_from_config_and_log():
    assert ev.strategy_labels(None, "v1") == ["v0", "v0b", "v1_v1"]
    paper = pd.DataFrame({"strategy": ["", "v0", "v1_v0x"]})
    assert ev.strategy_labels(paper, "v1") == ["v0", "v0b", "v1_v1", "v1_v0x"]


def test_cli_strategy_and_compare(tmp_path, capsys):
    path = tmp_path / "p.csv"
    _books().to_csv(path, index=False)
    ev.main(["--file", str(path), "--strategy", "v0b", "--min-trades", "2"])
    assert "strategy v0b" in capsys.readouterr().out
    ev.main(["--file", str(path), "--compare", "--events", str(tmp_path / "none.csv")])
    assert "| Verdict | **NOT YET** |" in capsys.readouterr().out


# --------------------------------------------------------------------------- event study


def _prices(closes, start="2026-10-12"):
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({"Open": [100.0] * len(closes), "Close": closes}, index=idx)


def test_event_outcomes_on_known_prices():
    # Mon 12 Oct onwards; release Mon 08:00 Stockholm (06:00 UTC) -> entry Mon open
    h = {"A": _prices([101, 102, 103, 104, 110, 111])}
    bench = _prices([100, 100, 100, 100, 101, 101])
    scored = pd.DataFrame(
        [
            {
                "release_id": "r",
                "ticker": "A",
                "published_at": "2026-10-12T06:00:00Z",
                "event_type": "report",
                "direction": "2",
                "surprise": "1",
                "material": "True",
                "confidence": "0.7",
            }
        ]
    )
    out = ev.event_outcomes(scored, h, bench, [1, 5, 20], cost=1.2)
    r = out.iloc[0]
    assert r["entry_date"] == "2026-10-12"
    assert r["ret_1d"] == pytest.approx(1.0) and r["abn_1d"] == pytest.approx(1.0 - 0 - 1.2)
    assert r["ret_5d"] == pytest.approx(10.0) and r["bench_5d"] == pytest.approx(1.0)
    assert r["abn_5d"] == pytest.approx(10.0 - 1.0 - 1.2)
    assert pd.isna(r["abn_20d"])  # not reached yet
    late = scored.assign(published_at="2026-10-12T08:00:00Z")  # 10:00 -> Tue open
    assert ev.event_outcomes(late, h, bench, [1], 0).iloc[0]["entry_date"] == "2026-10-13"


def _events(spec):
    """spec: list of (direction, abn_5d); published in order."""
    rows = []
    for i, (d, a) in enumerate(spec):
        rows.append(
            {
                "release_id": f"r{i}",
                "ticker": "A",
                "published_at": f"2026-10-{12 + i // 10:02d}T0{i % 10}:00:00Z",
                "entry_date": "",
                "event_type": "report" if d >= 0 else "guidance",
                "direction": str(d),
                "surprise": "1",
                "material": "True",
                "confidence": "0.5",
                "ret_1d": "",
                "bench_1d": "",
                "abn_1d": "",
                "ret_5d": "",
                "bench_5d": "",
                "abn_5d": str(a),
                "ret_20d": "",
                "bench_20d": "",
                "abn_20d": "",
            }
        )
    return pd.DataFrame(rows)


def test_event_study_groups_and_checks():
    spec = [(-2, -6), (-1, -2), (0, 0), (1, 2), (2, 6)] * 4
    spec = [(d, a + (i % 3) * 0.5) for i, (d, a) in enumerate(spec)]  # some noise
    res = ev.event_study(_events(spec), horizons=[1, 5, 20], min_events=10)
    g = res["groups"]["direction"]
    assert list(g) == ["-2", "-1", "0", "1", "2"]
    assert g["2"][5]["n"] == 4 and g["2"][5]["hit_rate"] == 100.0
    assert g["2"][5]["mean"] == pytest.approx(pd.Series([6.5, 6, 7, 6.5]).mean())
    assert res["monotonic"] and res["spread"] > 0
    assert res["spread_first_half"] > 0 and res["spread_second_half"] > 0 and res["spread_wo_best"] > 0
    assert res["verdict"] == "PASS" and set(res["groups"]["event_type"]) == {"guidance", "report"}


def test_event_study_not_yet_and_fail():
    spec = [(-2, -6), (2, 6)] * 3
    assert ev.event_study(_events(spec), min_events=300)["verdict"] == "NOT YET"
    backwards = [(-2, 5), (-1, 2), (1, -2), (2, -5)] * 3
    res = ev.event_study(_events(backwards), min_events=4)
    assert res["verdict"] == "FAIL" and not res["monotonic"] and res["spread"] < 0
    assert ev.event_study(None)["verdict"] == "NOT YET"


def test_event_study_one_lucky_event_fails_without_best():
    spec = [(1, -1), (1, -1), (1, 30), (-1, 0), (-1, 0), (-1, 0)]
    res = ev.event_study(_events(spec), min_events=6)
    assert res["spread"] > 0 and res["spread_wo_best"] < 0 and res["verdict"] == "FAIL"
