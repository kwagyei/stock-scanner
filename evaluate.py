#!/usr/bin/env python3
"""
Judge whether the scanner's signals are worth trading with real money.

Reads the closed paper trades and compares them, after costs, with the index over the
same periods. Read-only: it never changes any data.

Usage:
  python evaluate.py                   # main paper-trade rules
  python evaluate.py --variant         # the partial-exit variant instead
  python evaluate.py --cost 0.5 --stake 0.05 --min-trades 50

Verdict:
  NOT YET  fewer closed trades than --min-trades
  PASS     enough trades, average excess > 0, still > 0 without the best trade, t > 2
  FAIL     otherwise
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
PAPER_LOG = ROOT / "data" / "paper_trades.csv"
DEFAULTS = {"cost_pct": 1.0, "stake": 0.10, "min_trades": 30}
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
    return {**DEFAULTS, "move_threshold": move, **(cfg.get("evaluation") or {})}


def closed_trades(paper: pd.DataFrame, variant: bool = False) -> pd.DataFrame:
    """Closed trades as numbers: ret, bench (missing -> 0), reason, exit_date (oldest first)."""
    p = "pe_" if variant else ""
    if paper is None or paper.empty or f"{p}status" not in paper.columns:
        return pd.DataFrame(columns=["ticker", "ret", "bench", "reason", "exit_date", "move"])
    c = paper[paper[f"{p}status"] == "closed"]

    def col(name):
        return c[name] if name in c.columns else pd.Series("", index=c.index)

    out = pd.DataFrame({
        "ticker": col("ticker"),
        "ret": pd.to_numeric(col(f"{p}return_pct"), errors="coerce"),
        "bench": pd.to_numeric(col(f"{p}bench_return_pct"), errors="coerce").fillna(0.0),
        "reason": col(f"{p}exit_reason").replace("", "unknown"),
        "exit_date": col(f"{p}exit_date"),
        "move": pd.to_numeric(col("signal_move"), errors="coerce"),   # same signal for both rule sets
    }).dropna(subset=["ret"])
    return out.sort_values(["exit_date"], kind="stable").reset_index(drop=True)


def load_trades(path: Path = PAPER_LOG, variant: bool = False) -> pd.DataFrame | None:
    """None if the file does not exist."""
    if not Path(path).exists():
        return None
    return closed_trades(pd.read_csv(path, dtype=str).fillna(""), variant)


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


def evaluate(trades: pd.DataFrame | None, cost: float = DEFAULTS["cost_pct"],
             stake: float = DEFAULTS["stake"], min_trades: int = DEFAULTS["min_trades"],
             resamples: int = RESAMPLES, seed: int = SEED, move_threshold: float = 0.10) -> dict:
    """All statistics plus the verdict. Percent values are in percent (2.5 = 2.5%)."""
    res = {"n": 0, "cost": cost, "stake": stake, "min_trades": min_trades,
           "file_missing": trades is None}
    if trades is None or trades.empty:
        res.update(verdict="NOT YET", checks={})
        return res
    net = trades["ret"] - cost
    excess = net - trades["bench"]
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
    by_reason = (pd.DataFrame({"reason": trades["reason"], "net": net, "excess": excess})
                 .groupby("reason").agg(trades=("net", "size"), avg_net=("net", "mean"),
                                        avg_excess=("excess", "mean"))
                 .sort_values("trades", ascending=False))
    dirs = trades["move"].map(lambda m: direction(m, move_threshold)) if "move" in trades else "unknown"
    by_direction = (pd.DataFrame({"direction": dirs, "net": net, "excess": excess})
                    .groupby("direction").agg(trades=("net", "size"), win_rate=("net", lambda x: (x > 0).mean() * 100),
                                              avg_net=("net", "mean"), avg_excess=("excess", "mean"))
                    .reindex([d for d in DIRECTIONS if d in set(dirs)]))
    res.update({
        "n": n,
        "avg_net": float(net.mean()), "avg_bench": float(trades["bench"].mean()),
        "avg_excess": float(excess.mean()), "median_excess": float(excess.median()),
        "excess_wo_best": without(1), "excess_wo_best2": without(2),
        "win_rate": float((net > 0).mean() * 100), "n_wins": len(wins),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "losing_streak": _longest_losing_streak(net),
        "max_drawdown": _max_drawdown(net, stake),
        "t_stat": t, "p_positive": p_pos,
        "by_reason": by_reason, "by_direction": by_direction, "move_threshold": move_threshold,
    })
    checks = {
        "enough trades": n >= min_trades,
        "beats the index on average": res["avg_excess"] > 0,
        "still beats it without the best trade": (res["excess_wo_best"] or 0) > 0,
        "unlikely to be luck (t > 2)": t is not None and t > 2,
    }
    res["checks"] = checks
    res["verdict"] = "NOT YET" if n < min_trades else ("PASS" if all(checks.values()) else "FAIL")
    return res


def one_line(res: dict) -> str:
    """Short verdict for the digest, e.g. 'Evaluation: NOT YET – 12 of 30 trades'."""
    if res["verdict"] == "NOT YET":
        return f"Evaluation: NOT YET – {res['n']} of {res['min_trades']} trades"
    t = "–" if res["t_stat"] is None else f"{res['t_stat']:.1f}"
    return (f"Evaluation: {res['verdict']} – {res['n']} trades, "
            f"{res['avg_excess']:+.1f}% vs index per trade after costs, t = {t}")


def callout(res: dict) -> tuple[str, str]:
    """(GitHub callout kind, one-sentence explanation) for the verdict."""
    if res["verdict"] == "NOT YET":
        return "NOTE", ("Too early to judge. Keep the rules unchanged and let closed trades build up to "
                        f"{res['min_trades']}.")
    if res["verdict"] == "PASS":
        return "TIP", ("On paper the signals beat the index after costs, and it is unlikely to be luck. "
                       "This describes the past, not the future.")
    failed = [k for k, ok in res["checks"].items() if not ok]
    return "WARNING", ("On paper the signals do not reliably beat the index. Failed: " + ", ".join(failed) + ".")


def format_markdown(res: dict) -> str:
    """The full report as Markdown tables (GitHub pages). Same numbers as format_report."""
    if res["n"] == 0:
        return ("_No closed paper trades yet. The numbers appear here once the first trade closes._\n"
                if not res["file_missing"] else "_No paper-trade log yet. Run the scanner first._\n")
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
    out += ["", f"### Per trade, after {res['cost']:.1f}% cost", "", head + rows([
        ("Average return", _p(res["avg_net"]), ""),
        ("Index over the same days", _p(res["avg_bench"]), ""),
        ("Average excess over the index", _p(res["avg_excess"]), "the edge per trade; the number that matters"),
        ("Median excess", _p(res["median_excess"]), "a typical trade, ignoring extremes"),
        ("Excess without the best trade", _p(res["excess_wo_best"]), "does it rest on one lucky trade?"),
        ("Excess without the best two", _p(res["excess_wo_best2"]), ""),
    ]), "", "### Wins and losses", "", head + rows([
        ("Win rate", f"{res['win_rate']:.0f}%", f"{res['n_wins']} of {res['n']} trades made money"),
        ("Average win", "–" if res["avg_win"] is None else f"{res['avg_win']:+.1f}%", ""),
        ("Average loss", "–" if res["avg_loss"] is None else f"{res['avg_loss']:+.1f}%", ""),
        ("Longest losing streak", f"{res['losing_streak']} trade(s)", "losses in a row to sit through"),
        (f"Worst drawdown at {res['stake']:.0%} per trade", f"{res['max_drawdown']:+.1f}%",
         "biggest fall from a high point"),
    ]), "", "### Is it more than luck?", "", head + rows([
        ("t-statistic", t, "above 2 = unlikely to be chance alone"),
        ("Chance the true excess is above 0", pp, f"from {RESAMPLES:,} reshuffles of the trades"),
    ]), "", "### By exit reason", "", "| Exit | Trades | Avg return | Avg vs index |", "|:---|:---:|---:|---:|"]
    out += [f"| {reason} | {int(r['trades'])} | {r['avg_net']:+.1f}% | {r['avg_excess']:+.1f}% |"
            for reason, r in res["by_reason"].iterrows()]
    out += ["", "### By signal-day move", "",
            f"<sub>Did the trade follow a big drop, a big jump ({res['move_threshold']:.0%} or more) or no big move "
            "on the signal day?</sub>", "",
            "| Signal day | Trades | Win rate | Avg return | Avg vs index |", "|:---|:---:|:---:|---:|---:|"]
    out += [f"| {DIRECTIONS[d]} | {int(r['trades'])} | {r['win_rate']:.0f}% | {r['avg_net']:+.1f}% | "
            f"{r['avg_excess']:+.1f}% |" for d, r in res["by_direction"].iterrows()]
    return "\n".join(out) + "\n"


def _p(x, digits=1) -> str:
    return "–" if x is None else f"{x:+.{digits}f}%"


def format_report(res: dict, source: str = "data/paper_trades.csv", label: str = "main rules") -> str:
    lines = [f"Paper-trade evaluation ({label})",
             f"Source: {source} · cost {res['cost']:.2f}% per round trip · "
             f"{res['stake']:.0%} of the account per trade", ""]
    if res["file_missing"]:
        return "\n".join(lines + ["No paper-trade log found yet. Run the scanner first.", "",
                                  f"VERDICT: NOT YET – 0 of {res['min_trades']} trades"]) + "\n"
    if res["n"] == 0:
        return "\n".join(lines + ["No closed paper trades yet.", "",
                                  f"VERDICT: NOT YET – 0 of {res['min_trades']} trades"]) + "\n"

    mark = lambda ok: "✓" if ok else "✗"
    lines.append(f"VERDICT: {res['verdict']}" + (f" – {res['n']} of {res['min_trades']} trades so far"
                                                  if res["verdict"] == "NOT YET" else ""))
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
        lines.append(f"  {reason:<16} {int(r['trades']):>3} trade(s)   avg {r['avg_net']:+6.1f}%   "
                     f"vs index {r['avg_excess']:+6.1f}%")
    lines += ["", f"By signal-day move (big = {res['move_threshold']:.0%} or more)"]
    for d, r in res["by_direction"].iterrows():
        lines.append(f"  {DIRECTIONS[d]:<16} {int(r['trades']):>3} trade(s)   win {r['win_rate']:3.0f}%   "
                     f"avg {r['avg_net']:+6.1f}%   vs index {r['avg_excess']:+6.1f}%")
    lines += ["", "Paper results ignore spread and slippage beyond the cost above. Not investment advice."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    d = defaults_from_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(PAPER_LOG))
    ap.add_argument("--cost", type=float, default=d["cost_pct"],
                    help=f"round-trip cost in percent (default {d['cost_pct']})")
    ap.add_argument("--stake", type=float, default=d["stake"],
                    help=f"share of the account per trade, for the drawdown (default {d['stake']})")
    ap.add_argument("--min-trades", type=int, default=d["min_trades"],
                    help=f"closed trades needed before a verdict (default {d['min_trades']})")
    ap.add_argument("--variant", action="store_true", help="evaluate the partial-exit variant")
    ap.add_argument("--direction", choices=["drop", "jump", "flat"],
                    help="only trades whose signal day had a big drop / big jump / no big move")
    args = ap.parse_args(argv)

    trades = load_trades(Path(args.file), args.variant)
    if trades is not None and args.direction:
        trades = trades[trades["move"].map(lambda m: direction(m, d["move_threshold"])) == args.direction]
    res = evaluate(trades, args.cost, args.stake, args.min_trades, move_threshold=d["move_threshold"])
    label = ("partial-exit variant" if args.variant else "main rules") + (
        f", signal day: {DIRECTIONS[args.direction]}" if args.direction else "")
    path = Path(args.file).resolve()
    source = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    sys.stdout.write(format_report(res, source, label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
