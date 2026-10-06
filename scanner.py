#!/usr/bin/env python3
"""
Stockholm small-cap attention scanner (v1).

Research and alerting tool. It does NOT place trades.

Each run:
  1. Pulls daily price/volume history for every ticker in watchlist.csv (Yahoo Finance).
  2. Pulls recent news per company (Google News RSS, Swedish edition) and company
     releases straight from MFN (which also carries Cision-distributed releases).
  3. Scores each stock on: fresh company release, abnormal volume, big price move,
     news-count spike, small market cap.
  4. Flags "hype warnings": big 5-day run-ups with no company release behind them.
  5. Maintains a paper-trade log (entry at next open, fixed target/stop/time exit)
     and compares results with a benchmark index.
  6. Writes an HTML digest, appends CSV logs, optionally e-mails the digest.

Usage:
  python scanner.py                 # normal run
  python scanner.py --no-news       # skip news and release fetching (price/volume only)
  python scanner.py --no-email      # never send e-mail on this run
  python scanner.py --weekly        # also build the weekly summary now (otherwise Friday evening)
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import smtplib
import ssl
import sys
import time
import urllib.parse
from email.message import EmailMessage
from html import escape
from pathlib import Path

import feedparser
import pandas as pd
import yaml
import yfinance as yf

import evaluate as ev

try:  # python.org macOS builds ship without CA certificates; use certifi's bundle
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
SIGNALS_LOG = DATA_DIR / "signals_log.csv"
SCORES_LOG = DATA_DIR / "scores_log.csv"    # every stock, every run, with each score component
PAPER_LOG = DATA_DIR / "paper_trades.csv"
WEEKLY_LOG = DATA_DIR / "weekly_summary.csv"
DASHBOARD = ROOT / "DASHBOARD.md"
UTC = dt.timezone.utc
USER_AGENT = "Mozilla/5.0 (stock-attention-scanner)"

PAPER_COLUMNS = [
    "id", "ticker", "name", "signal_date", "score", "status",
    "entry_date", "entry_price", "bench_entry",
    "exit_date", "exit_price", "exit_reason", "bench_exit",
    "last_price", "return_pct", "bench_return_pct", "universe",
    # partial-exit variant (paper_trading.partial_exit); same entry as the main trade
    "pe_status", "pe_partial_date", "pe_partial_price",
    "pe_exit_date", "pe_exit_price", "pe_exit_reason", "pe_return_pct", "pe_bench_return_pct",
]
DEFAULT_UNIVERSE = "v1"   # trades from before watchlist versioning (the hand-picked list)


# --------------------------------------------------------------------------- config

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_watchlist(path: Path) -> pd.DataFrame:
    wl = pd.read_csv(path, comment="#", dtype=str).fillna("")
    wl["ticker"] = wl["ticker"].str.strip()
    wl["name"] = wl["name"].str.strip()
    if "query" not in wl.columns:
        wl["query"] = ""
    wl["query"] = wl.apply(lambda r: r["query"].strip() or r["name"], axis=1)
    if "mfn_slug" not in wl.columns:
        wl["mfn_slug"] = ""
    wl["mfn_slug"] = wl["mfn_slug"].str.strip()
    if "title_match" not in wl.columns:
        wl["title_match"] = ""
    wl["title_match"] = wl["title_match"].str.strip()
    return wl[wl["ticker"] != ""].reset_index(drop=True)


def watchlist_version(path: Path) -> str:
    """The '# version: …' comment written by build_watchlist.py; new paper trades carry it."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("# version:"):
                return line.split(":", 1)[1].strip() or DEFAULT_UNIVERSE
    return DEFAULT_UNIVERSE


# --------------------------------------------------------------------------- data

def fetch_history(ticker: str, period: str) -> pd.DataFrame:
    """Daily OHLCV. Index = tz-naive dates in exchange local time."""
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    except Exception as exc:  # network / symbol errors
        print(f"[warn] price fetch failed for {ticker}: {exc}", file=sys.stderr)
        return pd.DataFrame()
    if hist is None or hist.empty:
        return pd.DataFrame()
    hist = hist.dropna(subset=["Close"])
    idx = pd.to_datetime(hist.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    hist.index = idx.normalize()
    return hist


def fetch_market_cap(ticker: str) -> float | None:
    try:
        mc = getattr(yf.Ticker(ticker).fast_info, "market_cap", None)
        return float(mc) if mc else None
    except Exception:
        return None


def fetch_news(query: str, cfg: dict) -> list[dict]:
    ncfg = cfg["news"]
    q = f'"{query}" when:{ncfg["lookback_days"]}d'
    url = (
        "https://news.google.com/rss/search?q=" + urllib.parse.quote(q)
        + f"&hl={ncfg['language']}&gl={ncfg['country']}"
        + f"&ceid={ncfg['country']}:{ncfg['language']}"
    )
    try:
        feed = feedparser.parse(url, agent=USER_AGENT)
    except Exception as exc:
        print(f"[warn] news fetch failed for {query}: {exc}", file=sys.stderr)
        return []
    # feedparser swallows network/SSL errors into `bozo`; surface them instead of
    # silently reporting zero news.
    if feed.get("bozo") and not feed.entries:
        print(f"[warn] news fetch failed for {query}: {feed.get('bozo_exception')}", file=sys.stderr)
        return []
    items = []
    for e in feed.entries:
        pp = e.get("published_parsed")
        ts = dt.datetime(*pp[:6], tzinfo=UTC) if pp else None
        source = (e.get("source") or {}).get("title", "")
        items.append({"title": e.get("title", ""), "link": e.get("link", ""),
                      "source": source, "ts": ts})
    return items


def filter_titles(items: list[dict], word: str) -> list[dict]:
    """Keep news items whose title contains `word` as a whole word (case-sensitive),
    allowing the Swedish genitive -s ("Pricers vd").

    For names that are also common words, where Google News matches loosely.
    """
    if not word:
        return items
    pat = re.compile(rf"(?<!\w){re.escape(word)}s?(?!\w)")
    return [i for i in items if pat.search(i["title"])]


def fetch_releases(slug: str, cfg: dict) -> list[dict]:
    """Company releases from MFN's per-company feed (MFN- and Cision-distributed).

    Swedish and English versions of a release share a timestamp; keep one per
    timestamp, preferring the configured news language.
    """
    limit = cfg["releases"]["mfn_limit"]
    url = f"https://mfn.se/all/a/{urllib.parse.quote(slug)}.rss?limit={limit}"
    try:
        feed = feedparser.parse(url, agent=USER_AGENT)
    except Exception as exc:
        print(f"[warn] release fetch failed for {slug}: {exc}", file=sys.stderr)
        return []
    if feed.get("bozo") and not feed.entries:
        print(f"[warn] release fetch failed for {slug}: {feed.get('bozo_exception')}", file=sys.stderr)
        return []
    if not feed.entries:
        print(f"[warn] no MFN releases for '{slug}' – check mfn_slug in watchlist.csv", file=sys.stderr)
        return []
    lang = cfg["news"]["language"]
    by_ts = {}
    for e in feed.entries:
        pp = e.get("published_parsed")
        if not pp:
            continue
        ts = dt.datetime(*pp[:6], tzinfo=UTC)
        if ts not in by_ts or e.get("x_language") == lang:
            by_ts[ts] = {"title": e.get("title", ""), "link": e.get("link", ""),
                         "source": "MFN", "ts": ts}
    return sorted(by_ts.values(), key=lambda i: i["ts"], reverse=True)


# --------------------------------------------------------------------------- metrics

def price_metrics(hist: pd.DataFrame, cfg: dict) -> dict | None:
    n_base = cfg["volume_baseline_days"]
    if hist is None or len(hist) < 25:
        return None
    last, prev = hist.iloc[-1], hist.iloc[-2]
    base = hist["Volume"].iloc[-(n_base + 1):-1]
    avg_vol = float(base.mean()) if len(base) else 0.0
    vol_ratio = float(last["Volume"]) / avg_vol if avg_vol > 0 else float("nan")
    close_5 = hist["Close"].iloc[-6] if len(hist) >= 6 else hist["Close"].iloc[0]
    return {
        "date": hist.index[-1].date().isoformat(),
        "close": float(last["Close"]),
        "chg_1d": float(last["Close"] / prev["Close"] - 1),
        "chg_5d": float(last["Close"] / close_5 - 1),
        "volume": float(last["Volume"]),
        "vol_ratio": vol_ratio,
        "turnover_sek": float(last["Close"] * last["Volume"]),
    }


def news_metrics(items: list[dict], now: dt.datetime, cfg: dict, releases=()) -> dict:
    """items: Google News results. releases: direct company releases (MFN).

    Releases only feed the release signal and hype check; the news-spike count
    stays based on Google News alone.
    """
    lookback = cfg["news"]["lookback_days"]
    t = cfg["thresholds"]
    day = dt.timedelta(hours=24)
    dated = [i for i in items if i["ts"] is not None]
    recent = sorted([i for i in dated if now - i["ts"] <= day], key=lambda i: i["ts"], reverse=True)
    older = [i for i in dated if day < now - i["ts"] <= dt.timedelta(days=lookback)]
    baseline = len(older) / max(lookback - 1, 1)

    release_keys = [s.lower() for s in cfg["release_sources"]]
    ignore_keys = [s.lower() for s in cfg.get("release_ignore_titles") or []]
    def is_routine(i):
        return any(k in i["title"].lower() for k in ignore_keys)
    def is_release(i):
        return any(k in i["source"].lower() for k in release_keys) and not is_routine(i)

    releases = [r for r in releases if not is_routine(r)]

    direct_24h = [r for r in releases if now - r["ts"] <= day]
    direct_window = [r for r in releases if now - r["ts"] <= dt.timedelta(days=lookback)]
    releases_24h = sorted([i for i in recent if is_release(i)] + direct_24h,
                          key=lambda i: i["ts"], reverse=True)
    releases_window = [i for i in dated if is_release(i)] + direct_window
    spike = (len(recent) >= t["news_spike_min_items"]
             and len(recent) >= t["news_spike_multiplier"] * baseline)
    return {
        "n_24h": len(recent),
        "baseline_per_day": baseline,
        "spike": spike,
        "releases_24h": releases_24h,
        "has_release_window": bool(releases_window),
        "headlines": (releases_24h + [i for i in recent if not is_release(i)])[:3],
    }


SCORE_COMPONENTS = ["release", "volume", "price", "news", "small_cap"]


def score_breakdown(pm: dict, nm: dict, mcap: float | None, cfg: dict):
    """Points per component (SCORE_COMPONENTS), reasons and the hype flag."""
    t = cfg["thresholds"]
    pts, reasons = dict.fromkeys(SCORE_COMPONENTS, 0), []
    if nm["releases_24h"]:
        pts["release"] = 2
        reasons.append("company release <24h")
    vr = pm["vol_ratio"]
    if not math.isnan(vr):
        if vr >= t["volume_strong"]:
            pts["volume"] = 2
            reasons.append(f"volume {vr:.1f}x avg")
        elif vr >= t["volume_mild"]:
            pts["volume"] = 1
            reasons.append(f"volume {vr:.1f}x avg")
    if abs(pm["chg_1d"]) >= t["price_move_1d"]:
        pts["price"] = 1
        reasons.append(f"price {pm['chg_1d']:+.0%} 1d")
    if nm["spike"]:
        pts["news"] = 1
        reasons.append(f"news spike ({nm['n_24h']} in 24h vs {nm['baseline_per_day']:.1f}/day)")
    # Small cap only amplifies an existing signal; it is not a signal on its own.
    if sum(pts.values()) > 0 and mcap and mcap < t["small_cap_max_sek"]:
        pts["small_cap"] = 1
        reasons.append("small cap")
    hype = pm["chg_5d"] >= t["hype_5d_move"] and not nm["has_release_window"]
    return pts, reasons, hype


def score_stock(pm: dict, nm: dict, mcap: float | None, cfg: dict):
    pts, reasons, hype = score_breakdown(pm, nm, mcap, cfg)
    return sum(pts.values()), reasons, hype


# --------------------------------------------------------------------------- paper trading

def load_paper() -> pd.DataFrame:
    if PAPER_LOG.exists():
        df = pd.read_csv(PAPER_LOG, dtype=str).fillna("")
        for c in PAPER_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        df["universe"] = df["universe"].replace("", DEFAULT_UNIVERSE)
        return df[PAPER_COLUMNS]
    return pd.DataFrame(columns=PAPER_COLUMNS, dtype=str)


def _price_on(hist: pd.DataFrame, date: pd.Timestamp, field: str) -> float | None:
    if hist is None or hist.empty:
        return None
    sub = hist[hist.index <= date]
    if sub.empty:
        return None
    return float(sub.iloc[-1][field])


def update_paper(paper: pd.DataFrame, histories: dict, bench: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pc = cfg["paper_trading"]
    pe_enabled = (pc.get("partial_exit") or {}).get("enabled", False)
    for idx, row in paper.iterrows():
        main_todo = row["status"] in ("pending", "open")
        pe_todo = pe_enabled and row["status"] != "pending" and row["pe_status"] != "closed"
        if not (main_todo or pe_todo):
            continue
        h = histories.get(row["ticker"])
        if h is None or h.empty:
            continue
        if main_todo:
            _update_main(paper, idx, row, h, bench, pc)
        row = paper.loc[idx]
        if pe_enabled and row["entry_date"] and row["pe_status"] != "closed":
            _update_partial(paper, idx, row, h, bench, pc)
    return paper


def _update_main(paper, idx, row, h, bench, pc) -> None:
    """The fixed paper-trade rules: one exit at target, stop or time."""
    if row["status"] == "pending":
        after = h[h.index > pd.Timestamp(row["signal_date"])]
        if after.empty:
            return
        entry_day = after.index[0]
        paper.at[idx, "entry_date"] = entry_day.date().isoformat()
        paper.at[idx, "entry_price"] = f"{float(after.iloc[0]['Open']):.4f}"
        be = _price_on(bench, entry_day, "Open")
        paper.at[idx, "bench_entry"] = f"{be:.4f}" if be else ""
        paper.at[idx, "status"] = "open"
        row = paper.loc[idx]

    entry_day = pd.Timestamp(row["entry_date"])
    entry = float(row["entry_price"])
    target = entry * (1 + pc["take_profit"])
    stop = entry * (1 + pc["stop_loss"])
    bars = h[h.index >= entry_day]

    exit_info = None
    for n, (day, b) in enumerate(bars.iterrows(), start=1):
        gap_ok = n > 1  # on entry day we bought at the open, so no gap fill
        if b["Low"] <= stop:  # check stop first (conservative)
            exit_info = (day, min(b["Open"], stop) if gap_ok else stop, "stop")
        elif b["High"] >= target:
            exit_info = (day, max(b["Open"], target) if gap_ok else target, "target")
        elif n >= pc["max_hold_days"]:
            exit_info = (day, b["Close"], "time")
        if exit_info:
            break

    bench_entry = float(row["bench_entry"]) if row["bench_entry"] else None
    if exit_info:
        day, price, reason = exit_info
        paper.at[idx, "status"] = "closed"
        paper.at[idx, "exit_date"] = day.date().isoformat()
        paper.at[idx, "exit_price"] = f"{float(price):.4f}"
        paper.at[idx, "exit_reason"] = reason
        paper.at[idx, "last_price"] = f"{float(price):.4f}"
        paper.at[idx, "return_pct"] = f"{(float(price) / entry - 1) * 100:.2f}"
        bx = _price_on(bench, day, "Close")
    else:
        last = float(bars.iloc[-1]["Close"]) if not bars.empty else entry
        paper.at[idx, "last_price"] = f"{last:.4f}"
        paper.at[idx, "return_pct"] = f"{(last / entry - 1) * 100:.2f}"
        bx = _price_on(bench, bars.index[-1], "Close") if not bars.empty else None
    if bx and bench_entry:
        paper.at[idx, "bench_exit"] = f"{bx:.4f}"
        paper.at[idx, "bench_return_pct"] = f"{(bx / bench_entry - 1) * 100:.2f}"


def simulate_partial(bars: pd.DataFrame, entry: float, pc: dict) -> dict:
    """Partial-exit variant over the bars from the entry day on.

    Before the target it behaves like the main rules (stop first, gap fills at the open,
    time exit). At the target it sells `fraction`; the rest then exits at the remainder
    stop or at max_hold_days. Remainder rules apply from the next bar, since the order of
    high and low within a bar is unknown.
    """
    px = pc["partial_exit"]
    target = entry * (1 + pc["take_profit"])
    stop = entry * (1 + pc["stop_loss"])
    rem_stop = entry * (1 + px["remainder_stop"])
    partial = exit_ = None
    for n, (day, b) in enumerate(bars.iterrows(), start=1):
        gap_ok = n > 1
        if partial is None:
            if b["Low"] <= stop:
                exit_ = (day, min(b["Open"], stop) if gap_ok else stop, "stop")
            elif b["High"] >= target:
                partial = (day, max(b["Open"], target) if gap_ok else target)
                if n >= pc["max_hold_days"]:
                    exit_ = (day, b["Close"], "time")
            elif n >= pc["max_hold_days"]:
                exit_ = (day, b["Close"], "time")
        elif b["Low"] <= rem_stop:
            exit_ = (day, min(b["Open"], rem_stop), "remainder stop")
        elif n >= pc["max_hold_days"]:
            exit_ = (day, b["Close"], "time")
        if exit_:
            break
    end_day, end_px = (exit_[0], exit_[1]) if exit_ else (bars.index[-1], float(bars.iloc[-1]["Close"]))
    f = px["fraction"] if partial else 0.0
    ret = f * (partial[1] / entry - 1 if partial else 0) + (1 - f) * (float(end_px) / entry - 1)
    return {"status": "closed" if exit_ else "open", "partial": partial, "exit": exit_,
            "end_day": end_day, "ret": ret, "fraction": f}


def _update_partial(paper, idx, row, h, bench, pc) -> None:
    entry = float(row["entry_price"])
    bars = h[h.index >= pd.Timestamp(row["entry_date"])]
    if bars.empty:
        return
    r = simulate_partial(bars, entry, pc)
    paper.at[idx, "pe_status"] = r["status"]
    if r["partial"]:
        paper.at[idx, "pe_partial_date"] = r["partial"][0].date().isoformat()
        paper.at[idx, "pe_partial_price"] = f"{float(r['partial'][1]):.4f}"
    if r["exit"]:
        paper.at[idx, "pe_exit_date"] = r["exit"][0].date().isoformat()
        paper.at[idx, "pe_exit_price"] = f"{float(r['exit'][1]):.4f}"
        paper.at[idx, "pe_exit_reason"] = r["exit"][2]
    paper.at[idx, "pe_return_pct"] = f"{r['ret'] * 100:.2f}"
    # Index return weighted the same way as the position: part to the partial sale, rest to the end.
    b0 = float(row["bench_entry"]) if row["bench_entry"] else None
    b_end = _price_on(bench, r["end_day"], "Close")
    b_part = _price_on(bench, r["partial"][0], "Close") if r["partial"] else None
    if b0 and b_end and (b_part or not r["partial"]):
        f = r["fraction"]
        bret = f * ((b_part / b0 - 1) if r["partial"] else 0) + (1 - f) * (b_end / b0 - 1)
        paper.at[idx, "pe_bench_return_pct"] = f"{bret * 100:.2f}"


def add_paper_signals(paper: pd.DataFrame, rows: list[dict], cfg: dict,
                      universe: str = DEFAULT_UNIVERSE) -> pd.DataFrame:
    pc = cfg["paper_trading"]
    active = set(paper.loc[paper["status"].isin(["pending", "open"]), "ticker"])
    new = []
    for r in rows:
        if r["score"] >= pc["min_score"] and r["ticker"] not in active:
            rec = {c: "" for c in PAPER_COLUMNS}
            rec.update({
                "id": f"{r['ticker']}-{r['date']}",
                "ticker": r["ticker"], "name": r["name"],
                "signal_date": r["date"], "score": str(r["score"]),
                "status": "pending", "universe": universe,
            })
            new.append(rec)
            active.add(r["ticker"])
    if new:
        paper = pd.concat([paper, pd.DataFrame(new, columns=PAPER_COLUMNS)], ignore_index=True)
    return paper


# --------------------------------------------------------------------------- score log

def last_logged_bars() -> dict:
    """Latest bar date already logged per ticker (from the score log)."""
    if not SCORES_LOG.exists():
        return {}
    log = pd.read_csv(SCORES_LOG, dtype=str, usecols=lambda c: c in ("ticker", "status", "bar_date")).fillna("")
    log = log[log["status"] == "ok"]
    return log.groupby("ticker")["bar_date"].max().to_dict()


def score_log_rows(rows, missing, wl, now, universe, news_checked=True, stale=()) -> list[dict]:
    """One row per watchlist stock for this run, scored or not; stocks without price data too."""
    run = now.isoformat(timespec="minutes")
    out = []
    for r in rows:
        mc = r["market_cap"]
        out.append({
            "run_utc": run, "bar_date": r["date"], "ticker": r["ticker"], "name": r["name"],
            "status": "ok", "score": r["score"],
            **{f"pts_{k}": r["points"][k] for k in SCORE_COMPONENTS},
            "hype": r["hype"], "close": round(r["close"], 4), "chg_1d": round(r["chg_1d"], 4),
            "chg_5d": round(r["chg_5d"], 4),
            "vol_ratio": "" if math.isnan(r["vol_ratio"]) else round(r["vol_ratio"], 2),
            "news_24h": r["n_24h"] if news_checked else "", "releases_24h": r["n_releases"] if news_checked else "",
            "market_cap_msek": "" if not mc else round(mc / 1e6), "universe": universe,
            "reasons": "; ".join(r["reasons"]),
        })
    names = dict(zip(wl["ticker"], wl["name"]))
    blank = {c: "" for c in ["bar_date", "score", *(f"pts_{k}" for k in SCORE_COMPONENTS), "hype", "close",
                             "chg_1d", "chg_5d", "vol_ratio", "news_24h", "releases_24h", "market_cap_msek",
                             "reasons"]}
    out += [{"run_utc": run, "ticker": t, "name": names.get(t, ""),
             "status": "stale price data" if t in stale else "no price data",
             **blank, "universe": universe} for t in [*missing, *stale]]
    return out


def append_csv(path: Path, rows: list[dict]) -> None:
    """Append rows; if the columns differ from the file's header, rewrite it with all columns."""
    if not rows:
        return
    new = pd.DataFrame(rows)
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
        if header != list(new.columns):
            old = pd.read_csv(path, dtype=str).fillna("")
            pd.concat([old, new.astype(str)], ignore_index=True).to_csv(path, index=False)
            return
    new.to_csv(path, mode="a", index=False, header=not path.exists())


def paper_summary(paper: pd.DataFrame) -> dict:
    closed = paper[paper["status"] == "closed"]
    rets = pd.to_numeric(closed["return_pct"], errors="coerce").dropna()
    bench = pd.to_numeric(closed["bench_return_pct"], errors="coerce").dropna()
    return {
        "n_closed": len(closed),
        "n_open": int((paper["status"] == "open").sum()),
        "n_pending": int((paper["status"] == "pending").sum()),
        "win_rate": float((rets > 0).mean() * 100) if len(rets) else None,
        "avg_return": float(rets.mean()) if len(rets) else None,
        "avg_bench": float(bench.mean()) if len(bench) else None,
        **_pe_summary(paper),
    }


def _pe_summary(paper: pd.DataFrame) -> dict:
    closed = paper[paper["pe_status"] == "closed"]
    rets = pd.to_numeric(closed["pe_return_pct"], errors="coerce").dropna()
    bench = pd.to_numeric(closed["pe_bench_return_pct"], errors="coerce").dropna()
    return {"pe_n_closed": len(closed),
            "pe_avg_return": float(rets.mean()) if len(rets) else None,
            "pe_avg_bench": float(bench.mean()) if len(bench) else None}


# --------------------------------------------------------------------------- reporting

STYLE = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:16px;color:#1d1d1f;max-width:1100px}
h1{font-size:20px} h2{font-size:16px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid #eee;padding:6px;text-align:left;vertical-align:top}
td.n{text-align:right;white-space:nowrap} ul{margin:4px 0 0 16px;padding:0}
.warn{color:#a15c00} small{color:#666} .box{overflow-x:auto}
"""

def _fmt_pct(x, digits=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{digits}f}%"


def build_html(rows, missing, paper, summary, cfg, now_local, eval_line=None) -> str:
    alert_min = cfg["alert_min_score"]
    alerts = [r for r in rows if r["score"] >= alert_min]
    hype = [r for r in rows if r["hype"]]

    def stock_rows(rs):
        out = []
        for r in rs:
            heads = "".join(
                f'<li><a href="{escape(h["link"])}">{escape(h["title"])}</a></li>'
                for h in r["headlines"]
            )
            out.append(
                f"<tr><td><b>{escape(r['name'])}</b><br><small>{escape(r['ticker'])}</small></td>"
                f"<td class=n>{r['score']}</td>"
                f"<td class=n>{r['close']:.2f}</td>"
                f"<td class=n>{r['chg_1d']*100:+.1f}%</td>"
                f"<td class=n>{r['chg_5d']*100:+.1f}%</td>"
                f"<td class=n>{r['vol_ratio']:.1f}x</td>"
                f"<td class=n>{r['n_24h']}</td>"
                f"<td>{escape('; '.join(r['reasons'])) or '–'}"
                f"{'<ul>' + heads + '</ul>' if heads else ''}</td></tr>"
            )
        return "\n".join(out)

    head = ("<tr><th>Stock</th><th>Score</th><th>Close</th><th>1d</th><th>5d</th>"
            "<th>Vol/avg</th><th>News 24h</th><th>Why</th></tr>")

    paper_open = paper[paper["status"].isin(["open", "pending"])]
    paper_closed = paper[paper["status"] == "closed"].tail(15)

    def paper_rows(df, closed):
        out = []
        for _, p in df.iterrows():
            ret = pd.to_numeric(p["return_pct"], errors="coerce")
            bret = pd.to_numeric(p["bench_return_pct"], errors="coerce")
            out.append(
                f"<tr><td>{escape(p['name'])} <small>{escape(p['ticker'])}</small></td>"
                f"<td>{escape(p['status'])}</td><td>{escape(p['entry_date'])}</td>"
                f"<td class=n>{escape(p['entry_price'])}</td>"
                f"<td>{escape(p['exit_date'] if closed else '')}</td>"
                f"<td>{escape(p['exit_reason'] if closed else '')}</td>"
                f"<td class=n>{_fmt_pct(None if pd.isna(ret) else float(ret))}</td>"
                f"<td class=n>{_fmt_pct(None if pd.isna(bret) else float(bret))}</td></tr>"
            )
        return "\n".join(out) or "<tr><td colspan=8>None</td></tr>"

    phead = ("<tr><th>Stock</th><th>Status</th><th>Entry</th><th>Entry px</th>"
             "<th>Exit</th><th>Reason</th><th>Return</th><th>Index same period</th></tr>")

    s = summary
    summary_line = (
        f"Closed: {s['n_closed']} · Open: {s['n_open']} · Pending: {s['n_pending']}"
        + (f" · Win rate: {s['win_rate']:.0f}% · Avg return: {_fmt_pct(s['avg_return'])}"
           f" · Avg index return same periods: {_fmt_pct(s['avg_bench'])}"
           if s["n_closed"] else "")
        + (f"<br><small>Partial-exit variant: {s['pe_n_closed']} closed · avg {_fmt_pct(s['pe_avg_return'])}"
           f" vs index {_fmt_pct(s['pe_avg_bench'])}</small>" if s["pe_n_closed"] else "")
    )

    missing_html = (f"<p class=warn>No price data for: {escape(', '.join(missing))} "
                    f"– check the ticker symbols in watchlist.csv.</p>" if missing else "")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scanner digest {now_local:%Y-%m-%d %H:%M}</title>
<style>{STYLE}</style></head><body>
<h1>Attention scanner · {now_local:%a %d %b %Y, %H:%M}</h1>
<p><small>Research tool, not advice. Prices from Yahoo Finance may be delayed. News from Google News RSS.</small></p>
{missing_html}
<h2>Alerts (score ≥ {alert_min}) · {len(alerts)}</h2>
<div class=box><table>{head}{stock_rows(alerts) or '<tr><td colspan=8>No alerts.</td></tr>'}</table></div>
<h2>Hype warnings · {len(hype)}</h2>
<p><small>Up ≥ {cfg['thresholds']['hype_5d_move']:.0%} in 5 days with no company release found in the news window. Typical pump profile.</small></p>
<div class=box><table>{head}{stock_rows(hype) or '<tr><td colspan=8>None.</td></tr>'}</table></div>
<h2>Paper trading</h2>
{f"<p><b>{escape(eval_line)}</b> <small>(run <code>python evaluate.py</code> for the full report)</small></p>" if eval_line else ""}
<p>{summary_line}</p>
<p><small>Rules: enter at next open when score ≥ {cfg['paper_trading']['min_score']}; exit at +{cfg['paper_trading']['take_profit']:.0%}, {cfg['paper_trading']['stop_loss']:.0%}, or after {cfg['paper_trading']['max_hold_days']} trading days.</small></p>
<div class=box><table>{phead}{paper_rows(paper_open, False)}</table></div>
<p><small>Last closed trades</small></p>
<div class=box><table>{phead}{paper_rows(paper_closed, True)}</table></div>
<h2>Full watchlist</h2>
<div class=box><table>{head}{stock_rows(rows)}</table></div>
</body></html>"""


def build_markdown(rows, summary, cfg, now_local, eval_line=None) -> str:
    alerts = [r for r in rows if r["score"] >= cfg["alert_min_score"]]
    hype = [r for r in rows if r["hype"]]
    lines = [f"## Scanner {now_local:%Y-%m-%d %H:%M}", ""]
    lines.append(f"**Alerts:** {len(alerts)} · **Hype warnings:** {len(hype)}")
    lines.append("")
    if alerts:
        lines += ["| Stock | Score | 1d | Vol/avg | Why |", "|---|---|---|---|---|"]
        for r in alerts:
            lines.append(f"| {r['name']} ({r['ticker']}) | {r['score']} | {r['chg_1d']*100:+.1f}% "
                         f"| {r['vol_ratio']:.1f}x | {'; '.join(r['reasons'])} |")
    s = summary
    lines += ["", f"Paper trades: {s['n_closed']} closed, {s['n_open']} open, {s['n_pending']} pending"]
    if eval_line:
        lines += ["", f"**{eval_line}**"]
    if s["n_closed"]:
        lines.append(f"Win rate {s['win_rate']:.0f}%, avg {_fmt_pct(s['avg_return'])} "
                     f"vs index {_fmt_pct(s['avg_bench'])}")
    return "\n".join(lines) + "\n"


def send_email(html: str, subject: str) -> None:
    host = os.environ.get("SMTP_HOST")
    to = os.environ.get("EMAIL_TO")
    user = os.environ.get("SMTP_USER")
    pwd = os.environ.get("SMTP_PASS")
    port = int(os.environ.get("SMTP_PORT", "587"))
    if not (host and to and user and pwd):
        print("[info] e-mail skipped: SMTP_HOST/SMTP_USER/SMTP_PASS/EMAIL_TO not set")
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content("HTML digest attached as message body. Open in an HTML-capable mail client.")
    msg.add_alternative(html, subtype="html")
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(user, pwd)
        s.send_message(msg)
    print(f"[info] digest e-mailed to {to}")


# --------------------------------------------------------------------------- weekly summary

def week_label(d: dt.date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def weekly_due(now_local: dt.datetime, cfg: dict, force: bool = False) -> bool:
    """Once per ISO week, on the first run at/after the configured weekday and hour."""
    wc = cfg.get("weekly_summary") or {}
    if force:
        return True
    if not wc.get("enabled") or (now_local.weekday(), now_local.hour) < (wc["weekday"], wc["after_hour"]):
        return False
    if WEEKLY_LOG.exists():
        # Only a row generated at/after the scheduled time counts: an earlier --weekly
        # run must not stop the Friday summary.
        log = pd.read_csv(WEEKLY_LOG, dtype=str)
        gen = pd.to_datetime(log.loc[log["week"] == week_label(now_local.date()), "generated_local"])
        return not any((g.weekday(), g.hour) >= (wc["weekday"], wc["after_hour"]) for g in gen)
    return True


def load_signals() -> pd.DataFrame:
    """Signals log with one row per (ticker, bar date); several runs log the same bar."""
    cols = ["run_utc", "bar_date", "ticker", "name", "score", "reasons"]
    if not SIGNALS_LOG.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(SIGNALS_LOG, dtype=str).fillna("")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["score", "run_utc"]).drop_duplicates(["ticker", "bar_date"], keep="last")
    return df.sort_values(["bar_date", "ticker"]).reset_index(drop=True)


def signal_outcomes(signals: pd.DataFrame, histories: dict, bench: pd.DataFrame,
                    horizons: list[int]) -> pd.DataFrame:
    """Return after n trading days from the signal-day close, and the excess over the benchmark."""
    cols = [f"{k}_{n}d" for n in horizons for k in ("ret", "excess")]
    out = []
    for _, s in signals.iterrows():
        rec = {"ticker": s["ticker"], "name": s["name"], "bar_date": s["bar_date"],
               "score": int(s["score"]), "reasons": s.get("reasons", "")}
        h, day = histories.get(s["ticker"]), pd.Timestamp(s["bar_date"])
        if h is not None and not h.empty and day in h.index:
            pos = h.index.get_loc(day)
            c0, b0 = float(h["Close"].iloc[pos]), _price_on(bench, day, "Close")
            for n in horizons:
                if pos + n >= len(h):
                    continue
                r = float(h["Close"].iloc[pos + n]) / c0 - 1
                b1 = _price_on(bench, h.index[pos + n], "Close")
                rec[f"ret_{n}d"] = r
                rec[f"excess_{n}d"] = r - (b1 / b0 - 1) if b0 and b1 else float("nan")
        out.append(rec)
    columns = ["ticker", "name", "bar_date", "score", "reasons"] + cols
    if not out:  # keep text columns as text so date comparisons work on an empty window
        return pd.DataFrame(columns=columns, dtype=object)
    return pd.DataFrame(out).reindex(columns=columns)


def hit_rate_table(outcomes: pd.DataFrame, horizons: list[int], top_score: int) -> list[dict]:
    """Per score bucket (and overall): signals, hit rate vs benchmark, avg return, avg excess (in %)."""
    if outcomes.empty:
        return []
    o = outcomes.copy()
    o["bucket"] = o["score"].clip(upper=top_score).map(lambda s: f"{s}+" if s == top_score else str(s))
    rows = []
    for label, g in [("all", o)] + sorted(o.groupby("bucket")):
        row = {"score": label, "signals": len(g)}
        for n in horizons:
            ex, r = g[f"excess_{n}d"].dropna(), g[f"ret_{n}d"].dropna()
            row[f"n_{n}d"] = len(ex)
            row[f"hit_{n}d"] = float((ex > 0).mean() * 100) if len(ex) else None
            row[f"avg_{n}d"] = float(r.mean() * 100) if len(r) else None
            row[f"excess_{n}d"] = float(ex.mean() * 100) if len(ex) else None
        rows.append(row)
    return rows


def weekly_paper(paper: pd.DataFrame, week_start: dt.date, notional: float) -> dict:
    closed = paper[paper["status"] == "closed"]

    def stats(df):
        r = pd.to_numeric(df["return_pct"], errors="coerce")
        b = pd.to_numeric(df["bench_return_pct"], errors="coerce")
        ex = (r - b).dropna()
        r = r.dropna()
        return {
            "n": len(r),
            "win_rate": float((r > 0).mean() * 100) if len(r) else None,
            "avg_return": float(r.mean()) if len(r) else None,
            "avg_bench": float(b.dropna().mean()) if b.notna().any() else None,
            "avg_excess": float(ex.mean()) if len(ex) else None,
            "pnl_sek": float(r.sum() / 100 * notional),
        }

    week = closed[closed["exit_date"] >= week_start.isoformat()]
    opened = paper[paper["status"] == "open"]
    unreal = pd.to_numeric(opened["return_pct"], errors="coerce").dropna()
    by_universe = {u: stats(g) for u, g in closed.groupby("universe", sort=True)}
    pe = paper[paper["pe_status"] == "closed"].assign(
        return_pct=lambda d: d["pe_return_pct"], bench_return_pct=lambda d: d["pe_bench_return_pct"])
    pe_week = pe[pe["pe_exit_date"] >= week_start.isoformat()]
    return {"week": stats(week), "all": stats(closed), "by_universe": by_universe,
            "pe_week": stats(pe_week), "pe_all": stats(pe), "week_trades": week,
            "open": opened, "unrealised_sek": float(unreal.sum() / 100 * notional)}


def _fmt_rate(x):
    return "–" if x is None else f"{x:.0f}%"


def _fmt_sek(x):
    return f"{x:+,.0f} SEK".replace(",", " ")


def build_weekly(signals, outcomes, table, wp, cfg, now_local) -> tuple[str, str, dict]:
    """Returns (html, markdown, csv row)."""
    wc = cfg["weekly_summary"]
    horizons, label = wc["horizons"], week_label(now_local.date())
    week_start = now_local.date() - dt.timedelta(days=now_local.weekday())
    this_week = outcomes[outcomes["bar_date"] >= week_start.isoformat()]

    hhead = "<tr><th>Score</th><th>Signals</th>" + "".join(
        f"<th>{n}d: n</th><th>{n}d hit</th><th>{n}d avg</th><th>{n}d vs index</th>" for n in horizons) + "</tr>"
    hrows = "\n".join(
        f"<tr><td>{escape(r['score'])}</td><td class=n>{r['signals']}</td>" + "".join(
            f"<td class=n>{r[f'n_{n}d']}</td><td class=n>{_fmt_rate(r[f'hit_{n}d'])}</td>"
            f"<td class=n>{_fmt_pct(r[f'avg_{n}d'])}</td><td class=n>{_fmt_pct(r[f'excess_{n}d'])}</td>"
            for n in horizons) + "</tr>"
        for r in table) or f"<tr><td colspan={2 + 4 * len(horizons)}>No signals in the window.</td></tr>"

    def pct(v):
        return None if pd.isna(v) else float(v) * 100

    shead = "<tr><th>Stock</th><th>Bar date</th><th>Score</th>" + "".join(
        f"<th>{n}d</th>" for n in horizons) + "<th>Why</th></tr>"
    srows = "\n".join(
        f"<tr><td>{escape(s['name'])} <small>{escape(s['ticker'])}</small></td><td>{escape(s['bar_date'])}</td>"
        f"<td class=n>{s['score']}</td>"
        + "".join(f"<td class=n>{_fmt_pct(pct(s[f'ret_{n}d']))}</td>" for n in horizons)
        + f"<td>{escape(str(s['reasons']))}</td></tr>"
        for _, s in this_week.iterrows()) or f"<tr><td colspan={4 + len(horizons)}>No signals this week.</td></tr>"

    def pstats(st):
        return (f"{st['n']} closed · win rate {_fmt_rate(st['win_rate'])} · avg {_fmt_pct(st['avg_return'])}"
                f" vs index {_fmt_pct(st['avg_bench'])} (excess {_fmt_pct(st['avg_excess'])})"
                f" · P&amp;L {_fmt_sek(st['pnl_sek'])}")

    trows = "\n".join(
        f"<tr><td>{escape(t['name'])} <small>{escape(t['ticker'])}</small></td><td>{escape(t['entry_date'])}</td>"
        f"<td>{escape(t['exit_date'])}</td><td>{escape(t['exit_reason'])}</td>"
        f"<td class=n>{_fmt_pct(pd.to_numeric(t['return_pct'], errors='coerce'))}</td>"
        f"<td class=n>{_fmt_pct(pd.to_numeric(t['bench_return_pct'], errors='coerce'))}</td></tr>"
        for _, t in wp["week_trades"].iterrows()) or "<tr><td colspan=6>No trades closed this week.</td></tr>"
    notional = wc["notional_sek"]
    pe_cfg = cfg["paper_trading"].get("partial_exit") or {}
    pe_html = (f"<p><b>Partial-exit variant</b> <small>(sell {pe_cfg['fraction']:.0%} at the target, rest at "
               f"{pe_cfg['remainder_stop']:+.0%} stop or time; same entries)</small><br>"
               f"This week: {pstats(wp['pe_week'])}<br>All time: {pstats(wp['pe_all'])}</p>"
               if pe_cfg.get("enabled") else "")
    # Watchlist changes alter what the test measures, so show each watchlist version separately.
    urows = ("<p><small>All-time closed trades by watchlist version</small></p><div class=box><table>"
             "<tr><th>Version</th><th>Closed</th><th>Win rate</th><th>Avg</th><th>Index</th>"
             "<th>Excess</th><th>P&amp;L</th></tr>" + "".join(
                 f"<tr><td>{escape(u)}</td><td class=n>{st['n']}</td><td class=n>{_fmt_rate(st['win_rate'])}</td>"
                 f"<td class=n>{_fmt_pct(st['avg_return'])}</td><td class=n>{_fmt_pct(st['avg_bench'])}</td>"
                 f"<td class=n>{_fmt_pct(st['avg_excess'])}</td><td class=n>{_fmt_sek(st['pnl_sek'])}</td></tr>"
                 for u, st in wp["by_universe"].items()) + "</table></div>"
             if len(wp["by_universe"]) > 1 else "")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scanner weekly {label}</title>
<style>{STYLE}</style></head><body>
<h1>Attention scanner · weekly summary {label}</h1>
<p><small>Generated {now_local:%a %d %b %Y, %H:%M}. Research tool, not advice. Benchmark {escape(cfg['benchmark'])}.</small></p>
<h2>Signal hit rate · last {wc['lookback_weeks']} weeks</h2>
<p><small>Signals = score ≥ {cfg['alert_min_score']}, one per stock and bar date. Return is measured from the
signal-day close. A hit = the stock beat the index over the same days. n counts signals old enough to measure.</small></p>
<div class=box><table>{hhead}{hrows}</table></div>
<h2>Signals this week · {len(this_week)}</h2>
<div class=box><table>{shead}{srows}</table></div>
<h2>Paper trading</h2>
<p><b>This week:</b> {pstats(wp['week'])}</p>
<p><b>All time:</b> {pstats(wp['all'])}</p>
{pe_html}
<p><b>Open:</b> {len(wp['open'])} position(s), unrealised {_fmt_sek(wp['unrealised_sek'])}</p>
{urows}
<p><small>P&amp;L assumes {_fmt_sek(notional)[1:]} per trade, no costs.</small></p>
<div class=box><table><tr><th>Stock</th><th>Entry</th><th>Exit</th><th>Reason</th><th>Return</th><th>Index same period</th></tr>
{trows}</table></div>
</body></html>"""

    overall = table[0] if table else {}
    md = build_weekly_md(outcomes, table, wp, cfg, now_local)

    row = {"week": label, "generated_local": f"{now_local:%Y-%m-%d %H:%M}",
           "signals_week": len(this_week), "signals_window": len(outcomes)}
    for n in horizons:
        for k in ("n", "hit", "excess"):
            v = overall.get(f"{k}_{n}d")
            row[f"{k}_{n}d"] = "" if v is None else (v if k == "n" else round(v, 2))
    for scope in ("week", "all"):
        st = wp[scope]
        row.update({f"paper_{scope}_n": st["n"],
                    f"paper_{scope}_win_rate": "" if st["win_rate"] is None else round(st["win_rate"], 1),
                    f"paper_{scope}_avg_return": "" if st["avg_return"] is None else round(st["avg_return"], 2),
                    f"paper_{scope}_avg_bench": "" if st["avg_bench"] is None else round(st["avg_bench"], 2),
                    f"paper_{scope}_pnl_sek": round(st["pnl_sek"])})
    pe_all = wp["pe_all"]
    row.update({"paper_pe_all_n": pe_all["n"],
                "paper_pe_all_avg_return": "" if pe_all["avg_return"] is None else round(pe_all["avg_return"], 2),
                "paper_pe_all_pnl_sek": round(pe_all["pnl_sek"])})
    row.update({"open": len(wp["open"]), "unrealised_sek": round(wp["unrealised_sek"])})
    return html, md, row


def run_weekly(paper, histories, bench, cfg, now_local) -> tuple[str, str]:
    wc = cfg["weekly_summary"]
    week_start = now_local.date() - dt.timedelta(days=now_local.weekday())
    window_start = week_start - dt.timedelta(weeks=wc["lookback_weeks"] - 1)
    signals = load_signals()
    signals = signals[(signals["score"] >= cfg["alert_min_score"])
                      & (signals["bar_date"] >= window_start.isoformat())]
    outcomes = signal_outcomes(signals, histories, bench, wc["horizons"])
    table = hit_rate_table(outcomes, wc["horizons"], cfg["paper_trading"]["min_score"] + 1)
    wp = weekly_paper(paper, week_start, wc["notional_sek"])
    html, md, row = build_weekly(signals, outcomes, table, wp, cfg, now_local)

    (REPORT_DIR / f"weekly_{row['week']}.html").write_text(html, encoding="utf-8")
    (REPORT_DIR / "weekly_latest.html").write_text(html, encoding="utf-8")
    for name in (f"weekly_{row['week']}.md", "weekly_latest.md"):
        page = build_weekly_md(outcomes, table, wp, cfg, now_local, here=f"reports/{name}")
        (REPORT_DIR / name).write_text(page, encoding="utf-8")
    log = pd.read_csv(WEEKLY_LOG, dtype=str) if WEEKLY_LOG.exists() else pd.DataFrame()
    if not log.empty:
        log = log[log["week"] != row["week"]]   # a forced re-run replaces that week's row
    pd.concat([log, pd.DataFrame([row]).astype(str)], ignore_index=True).to_csv(WEEKLY_LOG, index=False)
    return html, md


# --------------------------------------------------------------------------- markdown pages
# A private repo can't show HTML, so the dashboard and the reports are also written as
# Markdown pages. They use only what GitHub renders: tables, emoji, callout boxes
# ("> [!NOTE]"), collapsible <details>, and badge images as buttons. Links are relative.

PAGES = [  # button label, colour, path from the repo root
    ("Dashboard", "24292f", "DASHBOARD.md"),
    ("Today", "0969da", "reports/latest.md"),
    ("Weekly", "8250df", "reports/weekly_latest.md"),
    ("Paper trades", "1a7f37", "reports/paper_trades.md"),
    ("Evaluation", "9a6700", "reports/evaluation.md"),
]
EXIT_LABELS = {"target": "🎯 Target", "stop": "🛑 Stop", "time": "⏱️ Time limit",
               "remainder stop": "🛑 Breakeven stop"}


def _md(text) -> str:
    """Escape text for a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _md_table(head: list[str], rows: list[list], empty: str = "None.", align: str = "") -> str:
    """align: one letter per column, l / c / r (default l)."""
    if not rows:
        return f"_{empty}_\n"
    seps = {"l": ":---", "c": ":---:", "r": "---:"}
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(seps[(align or "l" * len(head))[i]] for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def _repo_url() -> str | None:
    """Base URL of this repo when running in GitHub Actions (for links outside the files)."""
    server, repo = os.environ.get("GITHUB_SERVER_URL"), os.environ.get("GITHUB_REPOSITORY")
    return f"{server}/{repo}" if server and repo else None


def _button(label: str, colour: str, href: str) -> str:
    text = urllib.parse.quote(label.replace("-", "--").replace("_", "__"))
    return f"[![{label}](https://img.shields.io/badge/{text}-{colour}?style=for-the-badge)]({href})"


def _nav(here: str) -> str:
    """The button row. `here` = this page's path from the repo root (shown greyed out)."""
    up = "../" * here.count("/")
    btns = [_button(label, "d0d7de" if path == here else colour, up + path) for label, colour, path in PAGES]
    repo = _repo_url()
    if repo:
        btns.append(_button("Runs", "57606a", f"{repo}/actions"))
    btns.append(_button("Watchlist", "57606a", up + "watchlist.csv"))
    return " ".join(btns) + "\n"


def _callout(kind: str, text: str) -> str:
    """GitHub callout box. kind: NOTE, TIP, IMPORTANT, WARNING, CAUTION."""
    return f"> [!{kind}]\n> " + text.replace("\n", "\n> ") + "\n"


def _glance(cells: list[tuple[str, object, str]]) -> str:
    """One row of headline numbers: (label, value, small text below)."""
    return ("| " + " | ".join(c[0] for c in cells) + " |\n"
            + "|" + "|".join(":---:" for _ in cells) + "|\n"
            + "| " + " | ".join(f"**{v}**" + (f"<br><sub>{s}</sub>" if s else "") for _, v, s in cells) + " |\n")


def _arrow(pct, digits: int = 1) -> str:
    """Percent value (2.5 = 2.5%) as '▲ 2.5%' / '▼ 2.5%'."""
    if pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return "–"
    if round(abs(pct), digits) == 0:
        return f"{0:.{digits}f}%"
    return f"{'▲' if pct > 0 else '▼'} {abs(pct):.{digits}f}%"


def _num(x):
    v = pd.to_numeric(x, errors="coerce")
    return None if pd.isna(v) else float(v)


def _friendly(reason: str, short_price: bool = False) -> str:
    """Scanner reason text -> plain language. short_price: leave out the % (shown in its own column)."""
    r = reason.strip()
    if r.startswith("company release"):
        return "📣 Company release"
    if m := re.match(r"volume ([\d.]+)x avg", r):
        return f"📊 Volume {m[1]}× normal"
    if m := re.match(r"price ([+-]?\d+)% 1d", r):
        up = int(m[1]) > 0
        return f"{'📈 Big jump' if up else '📉 Big drop'}" + ("" if short_price else f" {_arrow(int(m[1]), 0)}")
    if m := re.match(r"news spike \((\d+) in 24h vs ([\d.]+)/day\)", r):
        return f"📰 {m[1]} news items today (usually {m[2]}/day)"
    if r == "small cap":
        return "🔹 Small cap"
    return _md(r)


def _why(reasons, short_price: bool = False) -> str:
    parts = reasons if isinstance(reasons, list) else [p for p in str(reasons).split(";") if p.strip()]
    return " · ".join(_friendly(p, short_price) for p in parts) or "–"


def _day(iso: str) -> str:
    """'2026-10-06' -> 'Tue 6 Oct' (keeps narrow table columns on one line)."""
    try:
        d = dt.date.fromisoformat(str(iso))
    except ValueError:
        return str(iso)
    return f"{d:%a}&nbsp;{d.day}&nbsp;{d:%b}"


def _stock(name, ticker) -> str:
    return f"**{_md(name)}**<br><sub>{ticker}</sub>"


COMPONENT_HEADS = {"release": ("📣", "Releases 24h"), "volume": ("📊", "Volume"), "price": ("📈", "Price today"),
                   "news": ("📰", "News 24h"), "small_cap": ("🔹", "Market cap")}
SCORE_LEGEND = ("<sub>**Bold** = counted towards the score: 📣 company release (+2) · 📊 volume 3× normal (+1) "
                "or 5× (+2) · 📈 price move ±10% (+1) · 📰 news spike (+1) · 🔹 under 5 bn SEK (+1, only "
                "with another signal).</sub>\n")


def _mcap(mc) -> str:
    if not mc:
        return "–"
    return f"{mc / 1e9:.1f}&nbsp;bn" if mc >= 1e9 else f"{mc / 1e6:.0f}&nbsp;M"


def _component_values(r) -> dict:
    """The measured value behind each score component, as table text."""
    vr = r["vol_ratio"]
    return {"release": str(r.get("n_releases", 0) or "–"),
            "volume": "–" if vr is None or math.isnan(vr) else f"{vr:.1f}×",
            "price": _arrow(r["chg_1d"] * 100).replace(" ", "&nbsp;"),
            "news": str(r.get("n_24h", 0) or "–"),
            "small_cap": _mcap(r.get("market_cap"))}


def _alerts_table(rs, empty) -> str:
    if not rs:
        return f"_{empty}_\n" if empty else ""
    heads = [f"{icon}<br><sub>{label}</sub>" for icon, label in COMPONENT_HEADS.values()]
    rows = []
    for r in rs:
        vals, pts = _component_values(r), r.get("points") or {}
        rows.append([_stock(r["name"], r["ticker"]), f"**{r['score']}**"]
                    + [f"**{vals[k]}**" if pts.get(k) else vals[k] for k in COMPONENT_HEADS]
                    + [_why(r["reasons"], short_price=True)])
    return (_md_table(["Stock", "Score", *heads, "Why it was flagged"], rows,
                      align="lc" + "r" * len(heads) + "l") + "\n" + SCORE_LEGEND)


def _headlines(rs) -> str:
    with_news = [r for r in rs if r["headlines"]]
    if not with_news:
        return ""
    n = sum(len(r["headlines"]) for r in with_news)
    out = [f"<details><summary>📰 Headlines behind these alerts ({n})</summary>", ""]
    for r in with_news:
        out.append(f"**{_md(r['name'])}**")
        out += [f"- [{_md(h['title'])}](<{h['link']}>)" for h in r["headlines"]]
        out.append("")
    return "\n".join(out + ["</details>", ""])


def _eval_callout(res: dict) -> str:
    kind, text = ev.callout(res)
    return _callout(kind, f"**{ev.one_line(res)}**<br>{text}")


def _paper_summary_table(scopes, cfg: dict) -> str:
    """scopes: (label, stats from weekly_paper, open count)."""
    notional = cfg["weekly_summary"]["notional_sek"]
    rows = []
    for label, st, n_open in scopes:
        rows.append([label, n_open if n_open is not None else "", st["n"], _fmt_rate(st["win_rate"]),
                     _arrow(st["avg_return"]), _arrow(st["avg_excess"]), _fmt_sek(st["pnl_sek"])])
    return (_md_table(["", "Open", "Closed", "Win rate", "Avg return", "Avg vs index", "Profit / loss"],
                      rows, align="lcccrrr")
            + f"\n<sub>Profit / loss as if every trade were {_fmt_sek(notional)[1:]}, no costs.</sub>\n")


def _open_table(paper: pd.DataFrame, cfg: dict, today: dt.date) -> str:
    pc = cfg["paper_trading"]
    pe_on = (pc.get("partial_exit") or {}).get("enabled")
    rows = []
    for _, p in paper[paper["status"] == "open"].iterrows():
        entry, ret, bench = _num(p["entry_price"]), _num(p["return_pct"]), _num(p["bench_return_pct"])
        day = min(len(pd.bdate_range(p["entry_date"], today)), pc["max_hold_days"])
        row = [_stock(p["name"], p["ticker"]), p["entry_date"],
               f"{entry:.2f} → {_num(p['last_price']) or entry:.2f}", _arrow(ret),
               _arrow(None if ret is None else ret - (bench or 0)),
               f"🎯 {entry * (1 + pc['take_profit']):.2f}<br>🛑 {entry * (1 + pc['stop_loss']):.2f}",
               f"{day} of {pc['max_hold_days']}"]
        if pe_on:
            row.append(f"half sold {p['pe_partial_date']}" if p["pe_partial_date"] else "–")
        rows.append(row)
    head = ["Stock", "Bought", "Price", "Return", "vs index", "Exit at", "Day"] + (["Partial-exit"] if pe_on else [])
    return _md_table(head, rows, "No open positions.", "llcrrlc" + ("l" if pe_on else ""))


def _pending_table(paper: pd.DataFrame) -> str:
    pend = paper[paper["status"] == "pending"]
    return _md_table(["Stock", "Signal", "Score", ""],
                     [[_stock(p["name"], p["ticker"]), _day(p["signal_date"]), p["score"], "⏳ buys at the next open"]
                      for _, p in pend.iterrows()], "Nothing waiting.", "llcl")


def _closed_table(df: pd.DataFrame, pe_on: bool, empty="No closed trades yet.") -> str:
    rows = []
    for _, p in df.iloc[::-1].iterrows():
        ret, bench = _num(p["return_pct"]), _num(p["bench_return_pct"])
        row = [_stock(p["name"], p["ticker"]), f"{p['entry_date']} → {p['exit_date']}",
               EXIT_LABELS.get(p["exit_reason"], p["exit_reason"]), _arrow(ret), _arrow(bench),
               _arrow(None if ret is None else ret - (bench or 0))]
        if pe_on:
            pe = _num(p["pe_return_pct"])
            row.append("still open" if p["pe_status"] == "open" else
                       f"{_arrow(pe)}<br><sub>{EXIT_LABELS.get(p['pe_exit_reason'], p['pe_exit_reason'])}</sub>"
                       if p["pe_status"] == "closed" else "–")
        rows.append(row)
    head = ["Stock", "Held", "Sold at", "Return", "Index", "vs index"] + (["Partial-exit"] if pe_on else [])
    return _md_table(head, rows, empty, "lllrrr" + ("r" if pe_on else ""))


def _updated(now_local, rows=None) -> str:
    bits = [f"Updated **{now_local:%a %d %b %Y, %H:%M}** (Stockholm)"]
    if rows:
        bits += [f"prices to {max(r['date'] for r in rows)}", f"{len(rows)} stocks"]
    repo, run_id = _repo_url(), os.environ.get("GITHUB_RUN_ID")
    if repo and run_id:
        bits.append(f"[run log]({repo}/actions/runs/{run_id})")
    return "<sub>" + " · ".join(bits) + "</sub>\n"


FOOTER = "\n---\n<sub>Research tool, not investment advice. Paper trades are hypothetical.</sub>\n"


def build_dashboard(rows, missing, paper, cfg, now_local, eval_res) -> str:
    alerts = [r for r in rows if r["score"] >= cfg["alert_min_score"]]
    hype = [r for r in rows if r["hype"]]
    pe_on = (cfg["paper_trading"].get("partial_exit") or {}).get("enabled")
    n_open, n_pend = int((paper["status"] == "open").sum()), int((paper["status"] == "pending").sum())
    n_closed = int((paper["status"] == "closed").sum())

    out = ["# 📊 Attention scanner", "", _nav("DASHBOARD.md"), _updated(now_local, rows)]
    if missing:
        out.append(_callout("WARNING", f"No price data for: {', '.join(missing)}. Check watchlist.csv."))
    out += [_glance([("🔔 Alerts", len(alerts), f"score ≥ {cfg['alert_min_score']}"),
                     ("⚠️ Hype warnings", len(hype), ""),
                     ("💼 Open trades", n_open, f"+{n_pend} waiting" if n_pend else ""),
                     ("✅ Closed trades", n_closed, ""),
                     ("⚖️ Evaluation", eval_res["verdict"],
                      f"{eval_res['n']} of {eval_res['min_trades']} trades" if eval_res["verdict"] == "NOT YET"
                      else f"{eval_res['n']} trades")]),
            "## 🔔 Today's alerts", "",
            _alerts_table(alerts, "✅ No alerts on this run."), _headlines(alerts)]
    if hype:
        out += ["## ⚠️ Hype warnings", "",
                _callout("CAUTION", f"Up {cfg['thresholds']['hype_5d_move']:.0%} or more in 5 days with no "
                                    "company release behind it. A typical pump profile."),
                _alerts_table(hype, "")]
    out += ["## 💼 Paper trades", "", _open_table(paper, cfg, now_local.date())]
    if n_pend:
        out += ["**Waiting to buy**", "", _pending_table(paper)]
    out += ["[All paper trades →](reports/paper_trades.md)", "", _eval_callout(eval_res),
            "[Full evaluation →](reports/evaluation.md) · "
            "[How to read it](README.md#judging-the-results-evaluatepy)", "", "## More", ""]

    sig = load_signals()
    sig = sig[sig["score"] >= cfg["alert_min_score"]]
    days = sorted(sig["bar_date"].unique())[-10:]
    sig = sig[sig["bar_date"].isin(days)].sort_values(["bar_date", "score"], ascending=False).head(30)
    out += [f"<details><summary>🗓️ Signals in the last 10 trading days ({len(sig)})</summary>", "",
            _md_table(["Date", "Stock", "Score", "Why"],
                      [[_day(s["bar_date"]), _stock(s["name"], s["ticker"]), s["score"], _why(s["reasons"])]
                       for _, s in sig.iterrows()], "No signals yet.", "llcl"),
            "</details>", ""]

    hist = []
    if WEEKLY_LOG.exists():
        for _, w in pd.read_csv(WEEKLY_LOG, dtype=str).fillna("").iloc[::-1].head(12).iterrows():
            page = REPORT_DIR / f"weekly_{w['week']}.md"
            week = f"[{w['week']}](reports/{page.name})" if page.exists() else w["week"]
            hit = lambda n: (f"{float(w[f'hit_{n}d']):.0f}% <sub>of {w[f'n_{n}d']}</sub>"
                             if w.get(f"hit_{n}d") else "–")
            hist.append([week, w["signals_week"], hit(5), hit(20), w["paper_all_n"],
                         _fmt_sek(float(w["paper_all_pnl_sek"] or 0))])
    out += ["<details><summary>📅 Weekly history</summary>", "",
            _md_table(["Week", "Signals", "Beat index after 5 days", "after 20 days", "Paper trades closed",
                       "Paper profit / loss"], hist, "No weekly summaries yet. The first one comes on Friday "
                       "evening.", "lccccr"),
            "</details>", "",
            "<details><summary>🗂️ Data files</summary>", "",
            "- [Signals log](data/signals_log.csv): every stock that scored or got a hype warning",
            "- [Score log](data/scores_log.csv): every stock, every run, with each score component",
            "- [Paper trades](data/paper_trades.csv): all trades, both rule sets",
            "- [Weekly summaries](data/weekly_summary.csv): one row per week",
            "- [Watchlist review](data/watchlist_review.csv): stocks the watchlist builder skipped",
            "- [Settings](config.yaml)", "", "</details>", FOOTER]
    return "\n".join(out)


def build_digest_md(rows, missing, cfg, now_local) -> str:
    alerts = [r for r in rows if r["score"] >= cfg["alert_min_score"]]
    hype = [r for r in rows if r["hype"]]
    out = ["# 📰 Today", "", _nav("reports/latest.md"), _updated(now_local, rows)]
    if missing:
        out.append(_callout("WARNING", f"No price data for: {', '.join(missing)}. Check watchlist.csv."))
    out += [_glance([("🔔 Alerts", len(alerts), f"score ≥ {cfg['alert_min_score']}"),
                     ("⚠️ Hype warnings", len(hype), ""),
                     ("📋 Stocks scanned", len(rows), f"{len(missing)} without data" if missing else "")]),
            "## 🔔 Alerts", "", _alerts_table(alerts, "✅ No alerts on this run."), _headlines(alerts),
            "## ⚠️ Hype warnings", "",
            f"<sub>Up {cfg['thresholds']['hype_5d_move']:.0%} or more in 5 days with no company release "
            "found. A typical pump profile.</sub>", "",
            _alerts_table(hype, "✅ None today."),
            "## 📋 All stocks", "",
            f"<details><summary>All {len(rows)} stocks, highest score first</summary>", "",
            _md_table(["Stock", "Score", "Close", "Today", "5 days", "Volume", "News", "Why"],
                      [[_stock(r["name"], r["ticker"]), r["score"], f"{r['close']:.2f}",
                        _arrow(r["chg_1d"] * 100), _arrow(r["chg_5d"] * 100),
                        "–" if math.isnan(r["vol_ratio"]) else f"{r['vol_ratio']:.1f}×", r["n_24h"],
                        _why(r["reasons"], short_price=True) if r["reasons"] else ""] for r in rows], "", "lcrrrrcl"),
            "</details>", FOOTER]
    return "\n".join(out)


def build_paper_md(paper, cfg, now_local, eval_res) -> str:
    pc = cfg["paper_trading"]
    pe = pc.get("partial_exit") or {}
    wp = weekly_paper(paper, now_local.date() - dt.timedelta(days=now_local.weekday()),
                      cfg["weekly_summary"]["notional_sek"])
    scopes = [("**Main rules**", wp["all"], int(paper["status"].isin(["open", "pending"]).sum()))]
    if pe.get("enabled"):
        scopes.append(("**Partial-exit variant**", wp["pe_all"], int((paper["pe_status"] == "open").sum())))
    closed = paper[paper["status"] == "closed"]
    out = ["# 💼 Paper trades", "", _nav("reports/paper_trades.md"), _updated(now_local),
           f"<sub>**Main rules:** buy at the next open when the score is {pc['min_score']} or more; sell at "
           f"+{pc['take_profit']:.0%} (🎯), {pc['stop_loss']:.0%} (🛑) or after {pc['max_hold_days']} trading days (⏱️)."
           + (f" **Partial-exit variant:** same buys; sell {pe['fraction']:.0%} at +{pc['take_profit']:.0%}, then "
              "the rest at the entry price or the time limit." if pe.get("enabled") else "") + "</sub>", "",
           "## All time", "", _paper_summary_table(scopes, cfg), _eval_callout(eval_res),
           f"## Open ({int((paper['status'] == 'open').sum())})", "", _open_table(paper, cfg, now_local.date()),
           f"## Waiting to buy ({int((paper['status'] == 'pending').sum())})", "", _pending_table(paper),
           f"## Closed ({len(closed)})", "", _closed_table(closed, pe.get("enabled"))]
    if len(wp["by_universe"]) > 1:
        out += ["<details><summary>By watchlist version</summary>", "",
                _md_table(["Version", "Closed", "Win rate", "Avg return", "Avg vs index", "Profit / loss"],
                          [[u, st["n"], _fmt_rate(st["win_rate"]), _arrow(st["avg_return"]),
                            _arrow(st["avg_excess"]), _fmt_sek(st["pnl_sek"])]
                           for u, st in wp["by_universe"].items()], align="lccrrr"),
                "</details>"]
    return "\n".join(out + [FOOTER])


def build_evaluation_md(res, pe_res, cfg, now_local) -> str:
    out = ["# ⚖️ Evaluation", "", _nav("reports/evaluation.md"), _updated(now_local),
           f"<sub>Do the closed paper trades beat {cfg['benchmark']} (OMX Stockholm 30) over the same days, "
           f"after {res['cost']:.1f}% cost per trade, by enough that it's unlikely to be luck? "
           "[How to read this](../README.md#judging-the-results-evaluatepy)</sub>", "",
           "## Main rules", "", _eval_callout(res), ev.format_markdown(res)]
    if pe_res is not None:
        out += ["## Partial-exit variant", "",
                "<details><summary>Same evaluation for the partial-exit variant</summary>", "",
                _eval_callout(pe_res), ev.format_markdown(pe_res), "</details>"]
    out.append("\n<sub>Recalculated on every run. A PASS describes the past, not the future.</sub>")
    return "\n".join(out + [FOOTER])


def build_weekly_md(outcomes, table, wp, cfg, now_local, here: str | None = None) -> str:
    """Weekly summary page. `here` adds the button row (left out of the GitHub job summary)."""
    wc = cfg["weekly_summary"]
    horizons, label = wc["horizons"], week_label(now_local.date())
    week_start = now_local.date() - dt.timedelta(days=now_local.weekday())
    this_week = outcomes[outcomes["bar_date"] >= week_start.isoformat()]
    pe_on = (cfg["paper_trading"].get("partial_exit") or {}).get("enabled")
    overall = table[0] if table else {}
    days = lambda n: f"{n} day" + ("s" if n != 1 else "")
    pct = lambda v: None if pd.isna(v) else float(v) * 100
    year, week, _ = now_local.date().isocalendar()

    out = [f"# 📅 Weekly summary · week {week}, {year}", ""]
    if here:
        out.append(_nav(here))
    out += [f"<sub>Generated **{now_local:%a %d %b %Y, %H:%M}** (Stockholm) · compared with "
            f"{cfg['benchmark']} (OMX Stockholm 30)</sub>", "",
            _glance([("🗓️ Signals this week", len(this_week), f"score ≥ {cfg['alert_min_score']}"),
                     ("💼 Paper trades closed", wp["week"]["n"], "this week"),
                     ("💰 Profit / loss", _fmt_sek(wp["week"]["pnl_sek"]), "this week")]
                    + ([("✂️ Partial-exit variant", _fmt_sek(wp["pe_week"]["pnl_sek"]), "this week")] if pe_on else [])
                    + [("📂 Open positions", len(wp["open"]), f"unrealised {_fmt_sek(wp['unrealised_sek'])}")]),
            "## 🎯 Did the signals beat the index?", "",
            f"<sub>Every signal from the last {wc['lookback_weeks']} weeks, measured from the close on the "
            "signal day. *Beat the index* = the stock did better than the index over the same days. "
            "Signals too recent to measure are not counted yet.</sub>", "",
            _md_table(["After", "Signals measured", "Beat the index", "Avg return", "Avg vs index"],
                      [[days(n), overall.get(f"n_{n}d", 0), _fmt_rate(overall.get(f"hit_{n}d")),
                        _arrow(overall.get(f"avg_{n}d")), _arrow(overall.get(f"excess_{n}d"))]
                       for n in horizons] if overall else [], "No signals in the window yet.", "lcccr")]
    if len(table) > 1:
        out += ["<details><summary>Split by score</summary>", "",
                _md_table(["Score", "Signals"] + [f"Beat index after {days(n)}" for n in horizons],
                          [[r["score"], r["signals"]] + [
                              f"{_fmt_rate(r[f'hit_{n}d'])} <sub>{_arrow(r[f'excess_{n}d'])} · {r[f'n_{n}d']} measured</sub>"
                              for n in horizons] for r in table[1:]], align="cc" + "c" * len(horizons)),
                "</details>", ""]
    out += [f"## 🗓️ This week's signals ({len(this_week)})", "",
            _md_table(["Stock", "Date", "Score"] + [f"After {days(n)}" for n in horizons] + ["Why"],
                      [[_stock(s["name"], s["ticker"]), _day(s["bar_date"]), s["score"]]
                       + [_arrow(pct(s[f"ret_{n}d"])) for n in horizons] + [_why(s["reasons"])]
                       for _, s in this_week.sort_values("bar_date").iterrows()],
                      "No signals this week.", "llc" + "r" * len(horizons) + "l"),
            "## 💼 Paper trading", ""]
    scopes = [("**Main rules**, this week", wp["week"], None), ("**Main rules**, all time", wp["all"], None)]
    if pe_on:
        scopes += [("**Partial-exit**, this week", wp["pe_week"], None),
                   ("**Partial-exit**, all time", wp["pe_all"], None)]
    out += [_md_table(["", "Closed", "Win rate", "Avg return", "Avg vs index", "Profit / loss"],
                      [[l, st["n"], _fmt_rate(st["win_rate"]), _arrow(st["avg_return"]),
                        _arrow(st["avg_excess"]), _fmt_sek(st["pnl_sek"])] for l, st, _ in scopes],
                      align="lccrrr"),
            "", f"<sub>Profit / loss as if every trade were {_fmt_sek(wc['notional_sek'])[1:]}, no costs.</sub>", "",
            "### Closed this week", "", _closed_table(wp["week_trades"], pe_on, "No trades closed this week.")]
    if len(wp["by_universe"]) > 1:
        out += ["<details><summary>By watchlist version (all time)</summary>", "",
                _md_table(["Version", "Closed", "Win rate", "Avg return", "Avg vs index", "Profit / loss"],
                          [[u, st["n"], _fmt_rate(st["win_rate"]), _arrow(st["avg_return"]),
                            _arrow(st["avg_excess"]), _fmt_sek(st["pnl_sek"])]
                           for u, st in wp["by_universe"].items()], align="lccrrr"),
                "</details>"]
    return "\n".join(out + [FOOTER])


# --------------------------------------------------------------------------- main

def _now() -> dt.datetime:
    return dt.datetime.now(UTC)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--watchlist", default=str(ROOT / "watchlist.csv"))
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--weekly", action="store_true", help="build the weekly summary on this run")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    wl = load_watchlist(Path(args.watchlist))
    DATA_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    now = _now()
    try:
        from zoneinfo import ZoneInfo
        now_local = now.astimezone(ZoneInfo("Europe/Stockholm"))
    except Exception:
        now_local = now

    paper = load_paper()
    active_tickers = set(paper.loc[paper["status"].isin(["pending", "open"])
                                   | (paper["pe_status"] == "open"), "ticker"])
    weekly = weekly_due(now_local, cfg, args.weekly)
    signal_tickers = set(load_signals()["ticker"]) if weekly else set()
    tickers = list(dict.fromkeys(list(wl["ticker"]) + sorted(active_tickers | signal_tickers)))

    print(f"[info] fetching prices for {len(tickers)} tickers …")
    histories = {t: fetch_history(t, cfg["history_period"]) for t in tickers}
    bench = fetch_history(cfg["benchmark"], cfg["history_period"])
    if bench.empty:
        print(f"[warn] no benchmark data for {cfg['benchmark']}", file=sys.stderr)

    rows, missing, stale = [], [], []
    seen = last_logged_bars()
    for _, w in wl.iterrows():
        pm = price_metrics(histories.get(w["ticker"]), cfg)
        if pm is None:
            missing.append(w["ticker"])
            continue
        # Yahoo sometimes drops the latest bar for a while (seen around midnight). Scoring an older
        # bar than one already logged would back-date signals and paper entries, so skip it.
        if pm["date"] < seen.get(w["ticker"], ""):
            print(f"[warn] {w['ticker']}: Yahoo returned {pm['date']}, older than the logged "
                  f"{seen[w['ticker']]}; skipped this run", file=sys.stderr)
            stale.append(w["ticker"])
            continue
        pause = cfg["news"].get("pause_seconds", 1.0)
        items, releases = [], []
        if not args.no_news:
            items = filter_titles(fetch_news(w["query"], cfg), w["title_match"])
            time.sleep(pause)
            if cfg.get("releases", {}).get("enabled") and w["mfn_slug"]:
                releases = fetch_releases(w["mfn_slug"], cfg)
                time.sleep(pause)
        nm = news_metrics(items, now, cfg, releases)
        mcap = fetch_market_cap(w["ticker"])
        pts, reasons, hype = score_breakdown(pm, nm, mcap, cfg)
        score = sum(pts.values())
        rows.append({
            "ticker": w["ticker"], "name": w["name"], **pm,
            "market_cap": mcap, "n_24h": nm["n_24h"],
            "headlines": nm["headlines"], "score": score,
            "reasons": reasons, "hype": hype, "points": pts, "n_releases": len(nm["releases_24h"]),
        })
    rows.sort(key=lambda r: (r["score"], r["vol_ratio"] if not math.isnan(r["vol_ratio"]) else 0),
              reverse=True)

    if cfg["paper_trading"]["enabled"]:
        paper = update_paper(paper, histories, bench, cfg)
        paper = add_paper_signals(paper, rows, cfg, watchlist_version(Path(args.watchlist)))
        paper.to_csv(PAPER_LOG, index=False)
    summary = paper_summary(paper)
    ec = {**ev.DEFAULTS, "digest_after_hour": 12, **(cfg.get("evaluation") or {})}
    eval_res = ev.evaluate(ev.closed_trades(paper), ec["cost_pct"], ec["stake"], ec["min_trades"])
    pe_res = (ev.evaluate(ev.closed_trades(paper, variant=True), ec["cost_pct"], ec["stake"], ec["min_trades"])
              if (cfg["paper_trading"].get("partial_exit") or {}).get("enabled") else None)
    eval_line = ev.one_line(eval_res)
    digest_eval = eval_line if now_local.hour >= ec["digest_after_hour"] else None

    log_rows = [{
        "run_utc": now.isoformat(timespec="minutes"), "bar_date": r["date"],
        "ticker": r["ticker"], "name": r["name"], "score": r["score"],
        "close": round(r["close"], 4), "chg_1d": round(r["chg_1d"], 4),
        "chg_5d": round(r["chg_5d"], 4),
        "vol_ratio": round(r["vol_ratio"], 2) if not math.isnan(r["vol_ratio"]) else "",
        "news_24h": r["n_24h"], "hype": r["hype"], "reasons": "; ".join(r["reasons"]),
    } for r in rows if r["score"] > 0 or r["hype"]]
    if log_rows:
        pd.DataFrame(log_rows).to_csv(SIGNALS_LOG, mode="a", index=False,
                                      header=not SIGNALS_LOG.exists())
    append_csv(SCORES_LOG, score_log_rows(rows, missing, wl, now, watchlist_version(Path(args.watchlist)),
                                          news_checked=not args.no_news, stale=stale))
    missing = missing + [f"{t} (stale)" for t in stale]   # shown in the reports' warnings

    html = build_html(rows, missing, paper, summary, cfg, now_local, digest_eval)
    stamp = f"{now_local:%Y-%m-%d_%H%M}"
    (REPORT_DIR / f"digest_{stamp}.html").write_text(html, encoding="utf-8")
    (REPORT_DIR / "latest.html").write_text(html, encoding="utf-8")

    md = build_markdown(rows, summary, cfg, now_local, digest_eval)
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)

    weekly_html = None
    if weekly:
        weekly_html, weekly_md = run_weekly(paper, histories, bench, cfg, now_local)
        if gh_summary:
            with open(gh_summary, "a", encoding="utf-8") as fh:
                fh.write("\n" + weekly_md)
        print(weekly_md)

    pages = {REPORT_DIR / "latest.md": build_digest_md(rows, missing, cfg, now_local),
             REPORT_DIR / "paper_trades.md": build_paper_md(paper, cfg, now_local, eval_res),
             REPORT_DIR / "evaluation.md": build_evaluation_md(eval_res, pe_res, cfg, now_local),
             DASHBOARD: build_dashboard(rows, missing, paper, cfg, now_local, eval_res)}
    for path, text in pages.items():
        path.write_text(text, encoding="utf-8")

    n_alerts = sum(r["score"] >= cfg["alert_min_score"] for r in rows)
    email_cfg = cfg.get("email", {})
    if email_cfg.get("enabled") and not args.no_email:
        if n_alerts or email_cfg.get("send_when", "always") == "always":
            send_email(html, f"Scanner {now_local:%d %b %H:%M}: {n_alerts} alert(s)")
        if weekly_html:
            send_email(weekly_html, f"Scanner weekly {week_label(now_local.date())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
