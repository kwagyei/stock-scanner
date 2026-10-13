#!/usr/bin/env python3
"""
Judge whether the scanner's signals are worth trading with real money.

Reads the closed paper trades and compares them, after costs, with the index over the
same periods. Read-only: it never changes any data.

Usage:
  python evaluate.py                   # v0 main paper-trade rules
  python evaluate.py --strategy v0b    # another paper book (v0 / v0b / v1_<prompt>)
  python evaluate.py --compare         # all strategies side by side + the LLM event study
  python evaluate.py --variant         # the partial-exit variant instead
  python evaluate.py --cost 0.5 --stake 0.05 --min-trades 50

Costs: costs.commission_pct per side plus half the bid-ask spread per side (the spread logged
at signal time, else costs.half_spread_pct). --cost replaces this with one round-trip figure.
Benchmark: the primary benchmark (small caps) where a trade has it, else the secondary (OMXS30),
always over exactly the trade's days. A trade with neither is left out and counted, not scored
against a 0% index.

Verdict:
  NOT YET  fewer closed trades than --min-trades
  PASS     enough trades, average excess > 0, still > 0 without the best trade, t > 2
  FAIL     otherwise
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
STATE_DIR = Path(os.environ.get("SCANNER_STATE_DIR") or ROOT)  # see scanner.STATE_DIR
PAPER_LOG = STATE_DIR / "data" / "paper_trades.csv"
EVENTS_LOG = STATE_DIR / "data" / "llm_events.csv"
DEFAULTS = {"cost_pct": 1.2, "stake": 0.10, "min_trades": 30}
COST_DEFAULTS = {"commission_pct": 0.1, "half_spread_pct": 0.5}
EVENT_DEFAULTS = {"min_events": 300, "horizons": [1, 5, 20]}
DIRECTIONS = {"drop": "▼ Drop", "jump": "▲ Jump", "flat": "– No big move", "unknown": "? Unknown"}


def direction(move, threshold: float = 0.10) -> str:
    """Signal-day price move (a fraction: -0.112 = -11.2%) -> drop / jump / flat / unknown.

    Uses the same threshold as the score's price component (thresholds.price_move_1d).
    """
    try:
        m = float(move)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(m):
        return "unknown"
    return "drop" if m <= -threshold else "jump" if m >= threshold else "flat"


RESAMPLES = 10_000
SEED = 42


def defaults_from_config(path: Path = ROOT / "config.yaml") -> dict:
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        cfg = {}
    move = ((cfg.get("thresholds") or {}).get("price_move_1d")) or 0.10
    costs = {**COST_DEFAULTS, **(cfg.get("costs") or {})}
    out = {
        **DEFAULTS,
        "move_threshold": move,
        **(cfg.get("evaluation") or {}),
        "costs": costs,
        **EVENT_DEFAULTS,
        **{k: v for k, v in (cfg.get("eval") or {}).items() if k in EVENT_DEFAULTS},
    }
    out["cost_pct"] = round_trip(costs)
    if (cfg.get("eval") or {}).get("min_trades"):
        out["min_trades"] = cfg["eval"]["min_trades"]
    return out


def round_trip(costs: dict) -> float:
    """Round-trip cost in percent: commission and half the spread, on both sides."""
    c = {**COST_DEFAULTS, **(costs or {})}
    return 2 * (c["commission_pct"] + c["half_spread_pct"])


def trade_costs(spread: pd.Series, costs: dict) -> pd.Series:
    """Per-trade round-trip cost: 2 x commission + the logged spread (= 2 x half spread),
    or 2 x costs.half_spread_pct where no spread was logged."""
    c = {**COST_DEFAULTS, **(costs or {})}
    return 2 * c["commission_pct"] + pd.to_numeric(spread, errors="coerce").fillna(2 * c["half_spread_pct"])


TRADE_COLUMNS = ["ticker", "ret", "bench", "bench_secondary", "reason", "exit_date", "move", "cost"]


def closed_trades(
    paper: pd.DataFrame, variant: bool = False, strategy: str | None = None, costs: dict | None = None
) -> pd.DataFrame:
    """Closed trades as numbers, oldest first: ret, bench (primary benchmark where known, else the
    secondary; NaN if neither is known), bench_secondary, reason, exit_date, move and cost (per-trade round
    trip from `costs`, or NaN = use evaluate's flat cost). `strategy` picks one paper book
    (rows without a strategy are v0). The partial-exit variant has only the secondary benchmark."""
    p = "pe_" if variant else ""
    if paper is None or paper.empty or f"{p}status" not in paper.columns:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    c = paper[paper[f"{p}status"] == "closed"]
    if strategy is not None:
        strat = c["strategy"].replace("", "v0") if "strategy" in c.columns else pd.Series("v0", index=c.index)
        c = c[strat == strategy]

    def col(name):
        return c[name] if name in c.columns else pd.Series("", index=c.index)

    secondary = pd.to_numeric(col(f"{p}bench_return_pct"), errors="coerce")
    primary = pd.to_numeric(col("bench_primary_return_pct"), errors="coerce") if not variant else secondary * np.nan
    out = pd.DataFrame(
        {
            "ticker": col("ticker"),
            "ret": pd.to_numeric(col(f"{p}return_pct"), errors="coerce"),
            "bench": primary.fillna(secondary),
            "bench_secondary": secondary,
            "reason": col(f"{p}exit_reason").replace("", "unknown"),
            "exit_date": col(f"{p}exit_date"),
            "move": pd.to_numeric(col("signal_move"), errors="coerce"),  # same signal for both rule sets
            "cost": trade_costs(col("spread_pct"), costs) if costs is not None else np.nan,
        }
    ).dropna(subset=["ret"])
    return out.sort_values(["exit_date"], kind="stable").reset_index(drop=True)


def load_trades(
    path: Path = PAPER_LOG, variant: bool = False, strategy: str | None = None, costs: dict | None = None
) -> pd.DataFrame | None:
    """None if the file does not exist."""
    if not Path(path).exists():
        return None
    return closed_trades(pd.read_csv(path, dtype=str).fillna(""), variant, strategy, costs)


def _longest_losing_streak(net: pd.Series) -> int:
    best = cur = 0
    for x in net:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return best


def _max_drawdown(net: pd.Series, stake: float) -> float:
    """Worst peak-to-trough fall of the account (in %), trades taken one after another,
    each with `stake` of the account."""
    equity = np.cumprod(1 + stake * net.to_numpy() / 100)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return float(((equity / peak) - 1).min() * 100) if len(equity) else 0.0


def evaluate(
    trades: pd.DataFrame | None,
    cost: float = DEFAULTS["cost_pct"],
    stake: float = DEFAULTS["stake"],
    min_trades: int = DEFAULTS["min_trades"],
    resamples: int = RESAMPLES,
    seed: int = SEED,
    move_threshold: float = 0.10,
) -> dict:
    """All statistics plus the verdict. Percent values are in percent (2.5 = 2.5%)."""
    res = {
        "n": 0,
        "cost": cost,
        "stake": stake,
        "min_trades": min_trades,
        "file_missing": trades is None,
        "n_no_bench": 0,
    }
    if trades is not None and not trades.empty and "bench" in trades:
        # A trade without any index return over its days can't be judged against the index;
        # counting it as an index return of 0% would quietly favour or penalise it. Leave it out.
        missing = trades["bench"].isna()
        res["n_no_bench"] = int(missing.sum())
        trades = trades[~missing]
    if trades is None or trades.empty:
        res.update(verdict="NOT YET", checks={})
        return res
    per_trade = trades["cost"] if "cost" in trades and trades["cost"].notna().all() else cost
    net = trades["ret"] - per_trade
    excess = net - trades["bench"]
    if "cost" in trades and trades["cost"].notna().all():
        res["cost"] = float(trades["cost"].mean())
    n = len(net)
    ranked = excess.sort_values(ascending=False)
    without = lambda k: float(ranked.iloc[k:].mean()) if n > k else None
    wins, losses = net[net > 0], net[net <= 0]
    sd = float(excess.std(ddof=1)) if n > 1 else float("nan")
    t = float(excess.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else None
    if n > 1:
        rng = np.random.default_rng(seed)
        boot = excess.to_numpy()[rng.integers(0, n, size=(resamples, n))].mean(axis=1)
        p_pos = float((boot > 0).mean())
    else:
        p_pos = None
    by_reason = (
        pd.DataFrame({"reason": trades["reason"], "net": net, "excess": excess})
        .groupby("reason")
        .agg(trades=("net", "size"), avg_net=("net", "mean"), avg_excess=("excess", "mean"))
        .sort_values("trades", ascending=False)
    )
    dirs = trades["move"].map(lambda m: direction(m, move_threshold)) if "move" in trades else "unknown"
    by_direction = (
        pd.DataFrame({"direction": dirs, "net": net, "excess": excess})
        .groupby("direction")
        .agg(
            trades=("net", "size"),
            win_rate=("net", lambda x: (x > 0).mean() * 100),
            avg_net=("net", "mean"),
            avg_excess=("excess", "mean"),
        )
        .reindex([d for d in DIRECTIONS if d in set(dirs)])
    )
    res.update(
        {
            "n": n,
            "avg_net": float(net.mean()),
            "avg_bench": float(trades["bench"].mean()),
            "avg_excess": float(excess.mean()),
            "median_excess": float(excess.median()),
            "avg_excess_secondary": float((net - trades["bench_secondary"]).dropna().mean())
            if "bench_secondary" in trades and trades["bench_secondary"].notna().any()
            else None,
            "excess_wo_best": without(1),
            "excess_wo_best2": without(2),
            "win_rate": float((net > 0).mean() * 100),
            "n_wins": len(wins),
            "avg_win": float(wins.mean()) if len(wins) else None,
            "avg_loss": float(losses.mean()) if len(losses) else None,
            "losing_streak": _longest_losing_streak(net),
            "max_drawdown": _max_drawdown(net, stake),
            "t_stat": t,
            "p_positive": p_pos,
            "by_reason": by_reason,
            "by_direction": by_direction,
            "move_threshold": move_threshold,
        }
    )
    checks = {
        "enough trades": n >= min_trades,
        "beats the index on average": res["avg_excess"] > 0,
        "still beats it without the best trade": (res["excess_wo_best"] or 0) > 0,
        "unlikely to be luck (t > 2)": t is not None and t > 2,
    }
    res["checks"] = checks
    res["verdict"] = "NOT YET" if n < min_trades else ("PASS" if all(checks.values()) else "FAIL")
    return res


# --------------------------------------------------------------------------- LLM event study

EVENT_KEYS = [
    "release_id",
    "ticker",
    "published_at",
    "entry_date",
    "event_type",
    "direction",
    "surprise",
    "material",
    "confidence",
]
GROUPS = ["direction", "surprise", "event_type"]


def event_columns(horizons) -> list[str]:
    return EVENT_KEYS + [f"{k}_{h}d" for h in horizons for k in ("ret", "bench", "abn")]


def event_outcomes(scored: pd.DataFrame, histories: dict, bench: pd.DataFrame, horizons, cost: float) -> pd.DataFrame:
    """Returns after each scored release: bought at the open of its entry day (strategies.entry_day_for),
    held to the close of the h-th trading day (h=1: the entry day's close). abn = return minus the
    benchmark over the same days, minus the round-trip `cost`; all in percent. Horizons not yet
    reached stay empty. `scored` = releases joined with usable LLM scores."""
    import strategies as st

    rows = []
    for _, e in scored.iterrows():
        rec = {k: e.get(k, "") for k in EVENT_KEYS}
        h = histories.get(e["ticker"])
        entry = st.entry_day_for(e["published_at"], h.index) if h is not None and not h.empty else None
        rec["entry_date"] = entry.date().isoformat() if entry is not None else ""
        if entry is not None:
            after = h[h.index >= entry]
            o = float(after.iloc[0]["Open"])
            b0 = float(bench.loc[entry, "Open"]) if bench is not None and entry in bench.index else None
            for n in horizons:
                if len(after) < n or not o:
                    continue
                day, c = after.index[n - 1], float(after.iloc[n - 1]["Close"])
                ret = (c / o - 1) * 100
                b1 = float(bench.loc[day, "Close"]) if b0 and day in bench.index else None
                rec[f"ret_{n}d"] = round(ret, 4)
                if b1:
                    rec[f"bench_{n}d"] = round((b1 / b0 - 1) * 100, 4)
                    rec[f"abn_{n}d"] = round(ret - rec[f"bench_{n}d"] - cost, 4)
        rows.append(rec)
    return pd.DataFrame(rows).reindex(columns=event_columns(horizons))


def _stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": float(x.mean()) if n else None,
        "median": float(x.median()) if n else None,
        "t": float(x.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else None,
        "hit_rate": float((x > 0).mean() * 100) if n else None,
    }


def _long_short(ev_: pd.DataFrame, col: str) -> float | None:
    d = pd.to_numeric(ev_["direction"], errors="coerce")
    up, down = ev_.loc[d > 0, col].dropna(), ev_.loc[d < 0, col].dropna()
    if up.empty or down.empty:
        return None
    return float(up.mean() - down.mean())


def event_study(events: pd.DataFrame | None, horizons=(1, 5, 20), min_events: int = 300, focus: int = 5) -> dict:
    """Abnormal returns grouped by direction / surprise / event_type at each horizon, plus the checks:
    returns rise with direction (monotonic), the up-minus-down spread is positive, holds in both
    halves of the sample and without the best event. NOT YET below `min_events` events with a
    `focus`-day outcome."""
    horizons = list(horizons)
    focus = focus if focus in horizons else horizons[len(horizons) // 2]
    res = {"horizons": horizons, "focus": focus, "min_events": min_events, "n": 0, "groups": {}, "checks": {}}
    if events is None or events.empty:
        res["verdict"] = "NOT YET"
        return res
    ev_ = events.copy()
    for h in horizons:
        ev_[f"abn_{h}d"] = pd.to_numeric(ev_[f"abn_{h}d"], errors="coerce")
    col = f"abn_{focus}d"
    done = ev_[ev_[col].notna()].sort_values("published_at", kind="stable")
    res["n"] = len(done)
    res["n_scored"] = len(ev_)
    for g in GROUPS:
        keys = ev_[g].astype(str)
        order = sorted(set(keys), key=lambda k: (float(k) if k.lstrip("-").isdigit() else 99, k))
        res["groups"][g] = {k: {h: _stats(ev_.loc[keys == k, f"abn_{h}d"]) for h in horizons} for k in order}
    means = [
        v[focus]["mean"]
        for k, v in sorted(res["groups"]["direction"].items(), key=lambda kv: int(kv[0]))
        if k.lstrip("-").isdigit() and v[focus]["n"]
    ]
    res["monotonic"] = len(means) >= 2 and all(a <= b for a, b in zip(means, means[1:]))
    res["spread"] = _long_short(done, col)
    half = len(done) // 2
    res["spread_first_half"] = _long_short(done.iloc[:half], col) if half else None
    res["spread_second_half"] = _long_short(done.iloc[half:], col) if half else None
    d = pd.to_numeric(done["direction"], errors="coerce")
    ups = done.loc[d > 0, col]
    wo_best = done.drop(ups.idxmax()) if not ups.empty else done
    res["spread_wo_best"] = _long_short(wo_best, col)
    up, down = done.loc[d > 0, col], done.loc[d < 0, col]
    if len(up) > 1 and len(down) > 1:
        se = np.sqrt(up.var(ddof=1) / len(up) + down.var(ddof=1) / len(down))
        res["spread_t"] = float((up.mean() - down.mean()) / se) if se > 0 else None
    else:
        res["spread_t"] = None
    pos = lambda x: x is not None and x > 0
    res["checks"] = {
        "enough events": res["n"] >= min_events,
        "returns rise with direction": res["monotonic"],
        "up minus down is positive": pos(res["spread"]),
        "holds in both halves": pos(res["spread_first_half"]) and pos(res["spread_second_half"]),
        "holds without the best event": pos(res["spread_wo_best"]),
        "unlikely to be luck (t > 2)": res["spread_t"] is not None and res["spread_t"] > 2,
    }
    res["verdict"] = "NOT YET" if res["n"] < min_events else "PASS" if all(res["checks"].values()) else "FAIL"
    return res


def load_events(path: Path = EVENTS_LOG) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def compare(paper: pd.DataFrame | None, events: pd.DataFrame | None, strategies: list[str], d: dict) -> dict:
    """Per-strategy verdicts (same rules and costs for each) and the event study."""
    out = {"strategies": {}, "min_trades": d["min_trades"]}
    for s in strategies:
        trades = closed_trades(paper, strategy=s, costs=d.get("costs")) if paper is not None else None
        out["strategies"][s] = evaluate(
            trades, d["cost_pct"], d["stake"], d["min_trades"], move_threshold=d["move_threshold"]
        )
        if paper is not None and not paper.empty:
            strat = paper["strategy"].replace("", "v0") if "strategy" in paper else pd.Series("v0", index=paper.index)
            book = paper[strat == s]
            out["strategies"][s]["n_open"] = int(book["status"].isin(["pending", "open"]).sum())
    out["events"] = event_study(events, d["horizons"], d["min_events"])
    return out


def strategy_labels(paper: pd.DataFrame | None, prompt_version: str | None = None) -> list[str]:
    """v0, v0b, v1_<prompt> in that order: from config, plus any other book in the paper log."""
    enabled = True
    if prompt_version is None:
        try:
            llm = (yaml.safe_load((ROOT / "config.yaml").read_text()) or {})["llm"]
            prompt_version, enabled = llm["prompt_version"], llm.get("enabled", True)
        except (OSError, KeyError, TypeError, yaml.YAMLError):
            prompt_version = "v1"
    labels = ["v0", "v0b"] + ([f"v1_{prompt_version}"] if enabled else [])
    if paper is not None and "strategy" in paper.columns:
        labels += sorted(set(paper["strategy"]) - set(labels) - {""})
    return labels


def format_compare_markdown(cmp: dict) -> str:
    """v0 / v0b / v1 side by side, then the event study, as Markdown."""
    names = list(cmp["strategies"])
    cell = lambda r, k, f: "–" if r.get(k) is None else f(r[k])
    rowdef = [
        ("Verdict", lambda r: f"**{r['verdict']}**"),
        ("Closed trades", lambda r: f"{r['n']} of {r['min_trades']}"),
        ("Open / pending", lambda r: str(r.get("n_open", 0))),
        ("Avg return after costs", lambda r: cell(r, "avg_net", _p)),
        ("Avg vs small caps", lambda r: cell(r, "avg_excess", _p)),
        ("Avg vs OMXS30", lambda r: cell(r, "avg_excess_secondary", _p)),
        ("Without the best trade", lambda r: cell(r, "excess_wo_best", _p)),
        ("Win rate", lambda r: cell(r, "win_rate", lambda x: f"{x:.0f}%")),
        ("t-statistic", lambda r: cell(r, "t_stat", lambda x: f"{x:.2f}")),
        ("Avg cost per trade", lambda r: f"{r['cost']:.2f}%"),
    ]
    out = ["| | " + " | ".join(f"`{n}`" for n in names) + " |", "|:---|" + "---:|" * len(names)]
    out += [f"| {label} | " + " | ".join(f(cmp["strategies"][n]) for n in names) + " |" for label, f in rowdef]
    e = cmp["events"]
    out += ["", "#### Event study: do the LLM scores predict returns?", ""]
    if not e.get("n_scored"):
        return "\n".join(out + ["_No scored releases yet._"]) + "\n"
    out.append(
        f"**{e['verdict']}** · {e['n']} of {e['min_events']} events with a {e['focus']}-day outcome "
        f"({e['n_scored']} scored). Abnormal return = return minus the small-cap index, after costs."
    )
    out += ["", "| | Check | Result |", "|:---:|:---|:---|"]
    det = {
        "enough events": f"{e['n']} (need {e['min_events']})",
        "returns rise with direction": "yes" if e.get("monotonic") else "no",
        "up minus down is positive": _p(e.get("spread")),
        "holds in both halves": f"{_p(e.get('spread_first_half'))} / {_p(e.get('spread_second_half'))}",
        "holds without the best event": _p(e.get("spread_wo_best")),
        "unlikely to be luck (t > 2)": "–" if e.get("spread_t") is None else f"t = {e['spread_t']:.2f}",
    }
    icon = lambda ok: "✅" if ok else ("⏳" if e["verdict"] == "NOT YET" else "❌")
    out += [f"| {icon(ok)} | {k[0].upper() + k[1:]} | {det[k]} |" for k, ok in e["checks"].items()]
    for g in GROUPS:
        out += [
            "",
            f"**By {g.replace('_', ' ')}**",
            "",
            "| " + g + " | " + " | ".join(f"N / mean / median / t / hit {h}d" for h in e["horizons"]) + " |",
            "|:---|" + ":---:|" * len(e["horizons"]),
        ]
        for k, by_h in e["groups"][g].items():
            cells = []
            for h in e["horizons"]:
                s_ = by_h[h]
                t = "–" if s_["t"] is None else f"{s_['t']:.1f}"
                cells.append(
                    "–"
                    if not s_["n"]
                    else f"{s_['n']} / {_p(s_['mean'])} / {_p(s_['median'])} / {t} / {s_['hit_rate']:.0f}%"
                )
            out.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def one_line(res: dict) -> str:
    """Short verdict for the digest, e.g. 'Evaluation: NOT YET – 12 of 30 trades'."""
    if res["verdict"] == "NOT YET":
        return f"Evaluation: NOT YET – {res['n']} of {res['min_trades']} trades"
    t = "–" if res["t_stat"] is None else f"{res['t_stat']:.1f}"
    return (
        f"Evaluation: {res['verdict']} – {res['n']} trades, "
        f"{res['avg_excess']:+.1f}% vs index per trade after costs, t = {t}"
    )


def callout(res: dict) -> tuple[str, str]:
    """(GitHub callout kind, one-sentence explanation) for the verdict."""
    if res["verdict"] == "NOT YET":
        return "NOTE", (
            f"Too early to judge. Keep the rules unchanged and let closed trades build up to {res['min_trades']}."
        )
    if res["verdict"] == "PASS":
        return "TIP", (
            "On paper the signals beat the index after costs, and it is unlikely to be luck. "
            "This describes the past, not the future."
        )
    failed = [k for k, ok in res["checks"].items() if not ok]
    return "WARNING", ("On paper the signals do not reliably beat the index. Failed: " + ", ".join(failed) + ".")


def format_markdown(res: dict) -> str:
    """The full report as Markdown tables (GitHub pages). Same numbers as format_report."""
    if res["n"] == 0:
        if res.get("n_no_bench"):
            return (
                f"_{res['n_no_bench']} closed trade(s), all left out: no index price for their days yet. "
                "They count once the index has those days._\n"
            )
        return (
            "_No closed paper trades yet. The numbers appear here once the first trade closes._\n"
            if not res["file_missing"]
            else "_No paper-trade log yet. Run the scanner first._\n"
        )
    t = "–" if res["t_stat"] is None else f"{res['t_stat']:.2f}"
    pp = "–" if res["p_positive"] is None else f"{res['p_positive']:.0%}"
    detail = {
        "enough trades": f"{res['n']} closed (need {res['min_trades']})",
        "beats the index on average": _p(res["avg_excess"]) + " per trade",
        "still beats it without the best trade": _p(res["excess_wo_best"]) + " per trade",
        "unlikely to be luck (t > 2)": f"t = {t}",
    }
    icon = lambda ok: "✅" if ok else ("⏳" if res["verdict"] == "NOT YET" else "❌")
    rows = lambda items: "\n".join(f"| {a} | **{b}** | {c} |" for a, b, c in items)
    head = "| | | |\n|:---|---:|:---|\n"
    out = ["### The four checks", "", "| | Check | Result |", "|:---:|:---|:---|"]
    out += [f"| {icon(ok)} | {name[0].upper() + name[1:]} | {detail[name]} |" for name, ok in res["checks"].items()]
    if res["verdict"] == "NOT YET":
        out += ["", "<sub>⏳ = not decided until there are enough trades. Early numbers can change a lot.</sub>"]
    if res.get("n_no_bench"):
        out += ["", f"<sub>{res['n_no_bench']} closed trade(s) left out: no index price for their days yet.</sub>"]
    out += [
        "",
        f"### Per trade, after {res['cost']:.1f}% cost",
        "",
        head
        + rows(
            [
                ("Average return", _p(res["avg_net"]), ""),
                ("Index over the same days", _p(res["avg_bench"]), ""),
                ("Average excess over the index", _p(res["avg_excess"]), "the edge per trade; the number that matters"),
                ("Median excess", _p(res["median_excess"]), "a typical trade, ignoring extremes"),
                ("Excess without the best trade", _p(res["excess_wo_best"]), "does it rest on one lucky trade?"),
                ("Excess without the best two", _p(res["excess_wo_best2"]), ""),
            ]
        ),
        "",
        "### Wins and losses",
        "",
        head
        + rows(
            [
                ("Win rate", f"{res['win_rate']:.0f}%", f"{res['n_wins']} of {res['n']} trades made money"),
                ("Average win", "–" if res["avg_win"] is None else f"{res['avg_win']:+.1f}%", ""),
                ("Average loss", "–" if res["avg_loss"] is None else f"{res['avg_loss']:+.1f}%", ""),
                ("Longest losing streak", f"{res['losing_streak']} trade(s)", "losses in a row to sit through"),
                (
                    f"Worst drawdown at {res['stake']:.0%} per trade",
                    f"{res['max_drawdown']:+.1f}%",
                    "biggest fall from a high point",
                ),
            ]
        ),
        "",
        "### Is it more than luck?",
        "",
        head
        + rows(
            [
                ("t-statistic", t, "above 2 = unlikely to be chance alone"),
                ("Chance the true excess is above 0", pp, f"from {RESAMPLES:,} reshuffles of the trades"),
            ]
        ),
        "",
        "### By exit reason",
        "",
        "| Exit | Trades | Avg return | Avg vs index |",
        "|:---|:---:|---:|---:|",
    ]
    out += [
        f"| {reason} | {int(r['trades'])} | {r['avg_net']:+.1f}% | {r['avg_excess']:+.1f}% |"
        for reason, r in res["by_reason"].iterrows()
    ]
    out += [
        "",
        "### By signal-day move",
        "",
        f"<sub>Did the trade follow a big drop, a big jump ({res['move_threshold']:.0%} or more) or no big move "
        "on the signal day?</sub>",
        "",
        "| Signal day | Trades | Win rate | Avg return | Avg vs index |",
        "|:---|:---:|:---:|---:|---:|",
    ]
    out += [
        f"| {DIRECTIONS[d]} | {int(r['trades'])} | {r['win_rate']:.0f}% | {r['avg_net']:+.1f}% | "
        f"{r['avg_excess']:+.1f}% |"
        for d, r in res["by_direction"].iterrows()
    ]
    return "\n".join(out) + "\n"


def _p(x, digits=1) -> str:
    return "–" if x is None else f"{x:+.{digits}f}%"


def format_report(res: dict, source: str = "data/paper_trades.csv", label: str = "main rules") -> str:
    lines = [
        f"Paper-trade evaluation ({label})",
        f"Source: {source} · cost {res['cost']:.2f}% per round trip · {res['stake']:.0%} of the account per trade",
        "",
    ]
    if res["file_missing"]:
        return (
            "\n".join(
                lines
                + [
                    "No paper-trade log found yet. Run the scanner first.",
                    "",
                    f"VERDICT: NOT YET – 0 of {res['min_trades']} trades",
                ]
            )
            + "\n"
        )
    if res["n"] == 0:
        what = (
            f"{res['n_no_bench']} closed trade(s), all left out: no index price for their days yet."
            if res.get("n_no_bench")
            else "No closed paper trades yet."
        )
        return "\n".join(lines + [what, "", f"VERDICT: NOT YET – 0 of {res['min_trades']} trades"]) + "\n"

    mark = lambda ok: "✓" if ok else "✗"
    lines.append(
        f"VERDICT: {res['verdict']}"
        + (f" – {res['n']} of {res['min_trades']} trades so far" if res["verdict"] == "NOT YET" else "")
    )
    detail = {
        "enough trades": f"{res['n']} (need {res['min_trades']})",
        "beats the index on average": _p(res["avg_excess"]) + " per trade",
        "still beats it without the best trade": _p(res["excess_wo_best"]) + " per trade",
        "unlikely to be luck (t > 2)": "t = " + ("–" if res["t_stat"] is None else f"{res['t_stat']:.2f}"),
    }
    for name, ok in res["checks"].items():
        lines.append(f"  {mark(ok)} {name}: {detail[name]}")
    if res["verdict"] == "NOT YET":
        lines.append("  (Too few trades to judge. The numbers below are early and can change a lot.)")
    if res.get("n_no_bench"):
        lines.append(f"  ({res['n_no_bench']} closed trade(s) left out: no index price for their days yet.)")
    t = "–" if res["t_stat"] is None else f"{res['t_stat']:.2f}"
    pp = "–" if res["p_positive"] is None else f"{res['p_positive']:.0%}"
    loss = "–" if res["avg_loss"] is None else f"{res['avg_loss']:+.1f}%"
    win = "–" if res["avg_win"] is None else f"{res['avg_win']:+.1f}%"
    lines += [
        "",
        f"Per trade, after costs ({res['n']} closed trades)",
        f"  Average return                     {_p(res['avg_net'])}",
        f"  Index over the same periods        {_p(res['avg_bench'])}",
        f"  Average excess over the index      {_p(res['avg_excess'])}   (median {_p(res['median_excess'])})",
        f"    ... without the best trade       {_p(res['excess_wo_best'])}",
        f"    ... without the best two         {_p(res['excess_wo_best2'])}",
        "  (If the excess disappears when one or two trades are removed, the result rests on luck.)",
        "",
        "Wins and losses",
        f"  Win rate                           {res['win_rate']:.0f}%  ({res['n_wins']} of {res['n']})",
        f"  Average win / average loss         {win} / {loss}",
        f"  Longest losing streak              {res['losing_streak']} trade(s)",
        f"  Worst drawdown at {res['stake']:.0%} per trade    {res['max_drawdown']:+.1f}%",
        "  (Drawdown = biggest fall from a high point, taking the trades one after another.)",
        "",
        "Is it more than luck?",
        f"  t-statistic of the excess          {t}   (above 2 = unlikely to be chance alone)",
        f"  Chance the true excess is above 0  {pp}   (bootstrap, {RESAMPLES:,} resamples)",
        "",
        "By exit reason",
    ]
    for reason, r in res["by_reason"].iterrows():
        lines.append(
            f"  {reason:<16} {int(r['trades']):>3} trade(s)   avg {r['avg_net']:+6.1f}%   "
            f"vs index {r['avg_excess']:+6.1f}%"
        )
    lines += ["", f"By signal-day move (big = {res['move_threshold']:.0%} or more)"]
    for d, r in res["by_direction"].iterrows():
        lines.append(
            f"  {DIRECTIONS[d]:<16} {int(r['trades']):>3} trade(s)   win {r['win_rate']:3.0f}%   "
            f"avg {r['avg_net']:+6.1f}%   vs index {r['avg_excess']:+6.1f}%"
        )
    lines += ["", "Paper results ignore spread and slippage beyond the cost above. Not investment advice."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    d = defaults_from_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(PAPER_LOG))
    ap.add_argument(
        "--cost", type=float, default=d["cost_pct"], help=f"round-trip cost in percent (default {d['cost_pct']})"
    )
    ap.add_argument(
        "--stake",
        type=float,
        default=d["stake"],
        help=f"share of the account per trade, for the drawdown (default {d['stake']})",
    )
    ap.add_argument(
        "--min-trades",
        type=int,
        default=d["min_trades"],
        help=f"closed trades needed before a verdict (default {d['min_trades']})",
    )
    ap.add_argument("--variant", action="store_true", help="evaluate the partial-exit variant")
    ap.add_argument(
        "--direction",
        choices=["drop", "jump", "flat"],
        help="only trades whose signal day had a big drop / big jump / no big move",
    )
    ap.add_argument("--strategy", default="v0", help="paper book: v0 (default), v0b or v1_<prompt version>")
    ap.add_argument(
        "--compare",
        action="store_true",
        help="all strategies side by side, plus the LLM event study (data/llm_events.csv)",
    )
    ap.add_argument("--events", default=str(EVENTS_LOG))
    args = ap.parse_args(argv)
    costs = None if "--cost" in (argv if argv is not None else sys.argv) else d["costs"]

    if args.compare:
        paper = pd.read_csv(args.file, dtype=str).fillna("") if Path(args.file).exists() else None
        if costs is None:  # --cost given: one flat round-trip cost for every trade
            d = {**d, "cost_pct": args.cost, "costs": None}
        labels = strategy_labels(paper)
        sys.stdout.write(
            format_compare_markdown(
                compare(
                    paper,
                    load_events(Path(args.events)),
                    labels,
                    {**d, "min_trades": args.min_trades, "stake": args.stake},
                )
            )
        )
        return 0

    trades = load_trades(Path(args.file), args.variant, args.strategy, costs)
    if trades is not None and args.direction:
        trades = trades[trades["move"].map(lambda m: direction(m, d["move_threshold"])) == args.direction]
    res = evaluate(trades, args.cost, args.stake, args.min_trades, move_threshold=d["move_threshold"])
    label = (
        ("partial-exit variant" if args.variant else "main rules")
        + (f", strategy {args.strategy}" if args.strategy != "v0" else "")
        + (f", signal day: {DIRECTIONS[args.direction]}" if args.direction else "")
    )
    path = Path(args.file).resolve()
    source = str(path.relative_to(STATE_DIR)) if path.is_relative_to(STATE_DIR) else str(path)
    sys.stdout.write(format_report(res, source, label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
