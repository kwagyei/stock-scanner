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
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
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

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
SIGNALS_LOG = DATA_DIR / "signals_log.csv"
PAPER_LOG = DATA_DIR / "paper_trades.csv"
UTC = dt.timezone.utc
USER_AGENT = "Mozilla/5.0 (stock-attention-scanner)"

PAPER_COLUMNS = [
    "id", "ticker", "name", "signal_date", "score", "status",
    "entry_date", "entry_price", "bench_entry",
    "exit_date", "exit_price", "exit_reason", "bench_exit",
    "last_price", "return_pct", "bench_return_pct",
]


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
    return wl[wl["ticker"] != ""].reset_index(drop=True)


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


def score_stock(pm: dict, nm: dict, mcap: float | None, cfg: dict):
    t = cfg["thresholds"]
    score, reasons = 0, []
    if nm["releases_24h"]:
        score += 2
        reasons.append("company release <24h")
    vr = pm["vol_ratio"]
    if not math.isnan(vr):
        if vr >= t["volume_strong"]:
            score += 2
            reasons.append(f"volume {vr:.1f}x avg")
        elif vr >= t["volume_mild"]:
            score += 1
            reasons.append(f"volume {vr:.1f}x avg")
    if abs(pm["chg_1d"]) >= t["price_move_1d"]:
        score += 1
        reasons.append(f"price {pm['chg_1d']:+.0%} 1d")
    if nm["spike"]:
        score += 1
        reasons.append(f"news spike ({nm['n_24h']} in 24h vs {nm['baseline_per_day']:.1f}/day)")
    # Small cap only amplifies an existing signal; it is not a signal on its own.
    if score > 0 and mcap and mcap < t["small_cap_max_sek"]:
        score += 1
        reasons.append("small cap")
    hype = pm["chg_5d"] >= t["hype_5d_move"] and not nm["has_release_window"]
    return score, reasons, hype


# --------------------------------------------------------------------------- paper trading

def load_paper() -> pd.DataFrame:
    if PAPER_LOG.exists():
        df = pd.read_csv(PAPER_LOG, dtype=str).fillna("")
        for c in PAPER_COLUMNS:
            if c not in df.columns:
                df[c] = ""
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
    for idx, row in paper.iterrows():
        if row["status"] not in ("pending", "open"):
            continue
        h = histories.get(row["ticker"])
        if h is None or h.empty:
            continue

        if row["status"] == "pending":
            after = h[h.index > pd.Timestamp(row["signal_date"])]
            if after.empty:
                continue
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
    return paper


def add_paper_signals(paper: pd.DataFrame, rows: list[dict], cfg: dict) -> pd.DataFrame:
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
                "status": "pending",
            })
            new.append(rec)
            active.add(r["ticker"])
    if new:
        paper = pd.concat([paper, pd.DataFrame(new, columns=PAPER_COLUMNS)], ignore_index=True)
    return paper


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
    }


# --------------------------------------------------------------------------- reporting

def _fmt_pct(x, digits=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{digits}f}%"


def build_html(rows, missing, paper, summary, cfg, now_local) -> str:
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
    )

    missing_html = (f"<p class=warn>No price data for: {escape(', '.join(missing))} "
                    f"– check the ticker symbols in watchlist.csv.</p>" if missing else "")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scanner digest {now_local:%Y-%m-%d %H:%M}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:16px;color:#1d1d1f;max-width:1100px}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border-bottom:1px solid #eee;padding:6px;text-align:left;vertical-align:top}}
td.n{{text-align:right;white-space:nowrap}} ul{{margin:4px 0 0 16px;padding:0}}
.warn{{color:#a15c00}} small{{color:#666}} .box{{overflow-x:auto}}
</style></head><body>
<h1>Attention scanner · {now_local:%a %d %b %Y, %H:%M}</h1>
<p><small>Research tool, not advice. Prices from Yahoo Finance may be delayed. News from Google News RSS.</small></p>
{missing_html}
<h2>Alerts (score ≥ {alert_min}) · {len(alerts)}</h2>
<div class=box><table>{head}{stock_rows(alerts) or '<tr><td colspan=8>No alerts.</td></tr>'}</table></div>
<h2>Hype warnings · {len(hype)}</h2>
<p><small>Up ≥ {cfg['thresholds']['hype_5d_move']:.0%} in 5 days with no company release found in the news window. Typical pump profile.</small></p>
<div class=box><table>{head}{stock_rows(hype) or '<tr><td colspan=8>None.</td></tr>'}</table></div>
<h2>Paper trading</h2>
<p>{summary_line}</p>
<p><small>Rules: enter at next open when score ≥ {cfg['paper_trading']['min_score']}; exit at +{cfg['paper_trading']['take_profit']:.0%}, {cfg['paper_trading']['stop_loss']:.0%}, or after {cfg['paper_trading']['max_hold_days']} trading days.</small></p>
<div class=box><table>{phead}{paper_rows(paper_open, False)}</table></div>
<p><small>Last closed trades</small></p>
<div class=box><table>{phead}{paper_rows(paper_closed, True)}</table></div>
<h2>Full watchlist</h2>
<div class=box><table>{head}{stock_rows(rows)}</table></div>
</body></html>"""


def build_markdown(rows, summary, cfg, now_local) -> str:
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


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--watchlist", default=str(ROOT / "watchlist.csv"))
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-email", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    wl = load_watchlist(Path(args.watchlist))
    DATA_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    now = dt.datetime.now(UTC)
    try:
        from zoneinfo import ZoneInfo
        now_local = now.astimezone(ZoneInfo("Europe/Stockholm"))
    except Exception:
        now_local = now

    paper = load_paper()
    active_tickers = set(paper.loc[paper["status"].isin(["pending", "open"]), "ticker"])
    tickers = list(dict.fromkeys(list(wl["ticker"]) + sorted(active_tickers)))

    print(f"[info] fetching prices for {len(tickers)} tickers …")
    histories = {t: fetch_history(t, cfg["history_period"]) for t in tickers}
    bench = fetch_history(cfg["benchmark"], cfg["history_period"])
    if bench.empty:
        print(f"[warn] no benchmark data for {cfg['benchmark']}", file=sys.stderr)

    rows, missing = [], []
    for _, w in wl.iterrows():
        pm = price_metrics(histories.get(w["ticker"]), cfg)
        if pm is None:
            missing.append(w["ticker"])
            continue
        pause = cfg["news"].get("pause_seconds", 1.0)
        items, releases = [], []
        if not args.no_news:
            items = fetch_news(w["query"], cfg)
            time.sleep(pause)
            if cfg.get("releases", {}).get("enabled") and w["mfn_slug"]:
                releases = fetch_releases(w["mfn_slug"], cfg)
                time.sleep(pause)
        nm = news_metrics(items, now, cfg, releases)
        mcap = fetch_market_cap(w["ticker"])
        score, reasons, hype = score_stock(pm, nm, mcap, cfg)
        rows.append({
            "ticker": w["ticker"], "name": w["name"], **pm,
            "market_cap": mcap, "n_24h": nm["n_24h"],
            "headlines": nm["headlines"], "score": score,
            "reasons": reasons, "hype": hype,
        })
    rows.sort(key=lambda r: (r["score"], r["vol_ratio"] if not math.isnan(r["vol_ratio"]) else 0),
              reverse=True)

    if cfg["paper_trading"]["enabled"]:
        paper = update_paper(paper, histories, bench, cfg)
        paper = add_paper_signals(paper, rows, cfg)
        paper.to_csv(PAPER_LOG, index=False)
    summary = paper_summary(paper)

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

    html = build_html(rows, missing, paper, summary, cfg, now_local)
    stamp = f"{now_local:%Y-%m-%d_%H%M}"
    (REPORT_DIR / f"digest_{stamp}.html").write_text(html, encoding="utf-8")
    (REPORT_DIR / "latest.html").write_text(html, encoding="utf-8")

    md = build_markdown(rows, summary, cfg, now_local)
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)

    n_alerts = sum(r["score"] >= cfg["alert_min_score"] for r in rows)
    email_cfg = cfg.get("email", {})
    if email_cfg.get("enabled") and not args.no_email:
        if n_alerts or email_cfg.get("send_when", "always") == "always":
            send_email(html, f"Scanner {now_local:%d %b %H:%M}: {n_alerts} alert(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
