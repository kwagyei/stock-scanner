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
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
PAPER_LOG = ROOT / "data" / "paper_trades.csv"
DEFAULTS = {"cost_pct": 1.0, "stake": 0.10, "min_trades": 30}
RESAMPLES = 10_000
SEED = 42


def defaults_from_config(path: Path = ROOT / "config.yaml") -> dict:
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        cfg = {}
    return {**DEFAULTS, **(cfg.get("evaluation") or {})}


def closed_trades(paper: pd.DataFrame, variant: bool = False) -> pd.DataFrame:
    """Closed trades as numbers: ret, bench (missing -> 0), reason, exit_date (oldest first)."""
    p = "pe_" if variant else ""
    if paper is None or paper.empty or f"{p}status" not in paper.columns:
        return pd.DataFrame(columns=["ticker", "ret", "bench", "reason", "exit_date"])
    c = paper[paper[f"{p}status"] == "closed"]

    def col(name):
        return c[name] if name in c.columns else pd.Series("", index=c.index)

    out = pd.DataFrame({
        "ticker": col("ticker"),
        "ret": pd.to_numeric(col(f"{p}return_pct"), errors="coerce"),
        "bench": pd.to_numeric(col(f"{p}bench_return_pct"), errors="coerce").fillna(0.0),
        "reason": col(f"{p}exit_reason").replace("", "unknown"),
        "exit_date": col(f"{p}exit_date"),
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
             resamples: int = RESAMPLES, seed: int = SEED) -> dict:
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
        "by_reason": by_reason,
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
    args = ap.parse_args(argv)

    trades = load_trades(Path(args.file), args.variant)
    res = evaluate(trades, args.cost, args.stake, args.min_trades)
    label = "partial-exit variant" if args.variant else "main rules"
    path = Path(args.file).resolve()
    source = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    sys.stdout.write(format_report(res, source, label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
