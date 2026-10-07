#!/usr/bin/env python3
"""
Fund risk news monitor.

Scans Google News daily for anything that could hurt a fund: its holdings, their
customers and suppliers, geopolitics, trade restrictions, natural disasters,
infrastructure, labour, legal issues, market stress. Flags and ranks the issues.

Only NEW items are reported each run (a seen-store removes repeats).
Research and alerting only. No trading.

Usage:
  python fund_watch.py              # normal run
  python fund_watch.py --no-email
  python fund_watch.py --all        # ignore seen-store (show everything in window); writes no log
  python fund_watch.py --test-email # send this run's report even if nothing is flagged (checks SMTP)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
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
import yaml

try:  # python.org macOS builds ship without CA certificates; use certifi's bundle
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
SEEN_FILE = DATA_DIR / "fund_watch_seen.json"
LOG_FILE = DATA_DIR / "fund_watch_log.csv"
UTC = dt.timezone.utc
SEVERITY_LABEL = {3: "HIGH", 2: "MEDIUM", 1: "LOW"}


# --------------------------------------------------------------------------- config / queries

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def holding_names(h: dict) -> list[str]:
    return [h["name"], *h.get("aliases", [])]


def build_queries(cfg: dict) -> list[tuple[str, str]]:
    """Returns (group, query) pairs."""
    q = []
    for name in cfg["fund"]["search_names"]:
        q.append(("fund", f'"{name}"'))
    for h in cfg["holdings"]:
        names = " OR ".join(f'"{n}"' for n in holding_names(h))
        q.append(("holding", names))
    deps = cfg["dependencies"]
    # Customers/suppliers only matter in a Taiwan/chip context; keep queries focused.
    for c in deps["customers"]:
        q.append(("customer", f'"{c}" (TSMC OR Taiwan OR chips OR orders)'))
    for s in deps["suppliers"]:
        q.append(("supplier", f'"{s}" (Taiwan OR TSMC OR China OR export)'))
    for t in cfg["topics"]:
        q.append(("topic", t))
    return q


def compile_categories(cfg: dict) -> dict:
    out = {}
    for cat, spec in cfg["risk_categories"].items():
        pats = [re.compile(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])", re.IGNORECASE)
                for k in spec["keywords"]]
        out[cat] = (spec["severity"], spec["keywords"], pats)
    return out


def compile_names(names: list[str]) -> list[re.Pattern]:
    return [re.compile(r"(?<!\w)" + re.escape(n) + r"(?!\w)", re.IGNORECASE) for n in names]


# --------------------------------------------------------------------------- fetching

def fetch_feed(query: str, cfg: dict) -> list[dict]:
    n = cfg["news"]
    lang, country = n["language"], n["country"]
    url = ("https://news.google.com/rss/search?q="
           + urllib.parse.quote(f'{query} when:{n["lookback"]}')
           + f"&hl={lang}&gl={country}&ceid={country}:{lang.split('-')[0]}")
    try:
        feed = feedparser.parse(url, agent="Mozilla/5.0 (fund-risk-monitor)")
    except Exception as exc:
        print(f"[warn] feed failed for {query!r}: {exc}", file=sys.stderr)
        return []
    if feed.get("bozo") and not feed.entries:  # feedparser hides network/parse errors here
        print(f"[warn] feed failed for {query!r}: {feed.get('bozo_exception')}", file=sys.stderr)
        return []
    items = []
    for e in feed.entries:
        pp = e.get("published_parsed")
        items.append({
            "title": e.get("title", "").strip(),
            "link": e.get("link", ""),
            "source": (e.get("source") or {}).get("title", ""),
            "ts": dt.datetime(*pp[:6], tzinfo=UTC) if pp else None,
            "summary": re.sub(r"<[^>]+>", " ", e.get("summary", "")),
        })
    return items


def item_key(title: str) -> str:
    # Google News titles end with " - Source"; drop it so the same story from
    # several outlets with identical headlines dedups.
    core = re.sub(r"\s+-\s+[^-]+$", "", title).lower()
    core = re.sub(r"[^a-z0-9 ]+", "", core)
    return hashlib.sha1(" ".join(core.split()).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- classification

def classify(item: dict, cats: dict, name_sets: dict, cfg: dict) -> dict | None:
    text = f"{item['title']} {item['summary']}"
    hits, max_sev = {}, 0
    for cat, (sev, kws, pats) in cats.items():
        matched = [kw for kw, p in zip(kws, pats) if p.search(text)]
        if matched:
            hits[cat] = matched
            max_sev = max(max_sev, sev)
    if not hits:
        return None

    s = cfg["scoring"]
    score = max_sev
    tags = []
    top = [n for n, p in name_sets["top"] if p.search(text)]
    other = [n for n, p in name_sets["other"] if p.search(text)]
    deps = [n for n, p in name_sets["deps"] if p.search(text)]
    if top:
        score += s["holding_bonus"]
        tags += top
    elif other:
        score += s["other_holding_bonus"]
        tags += other
    if deps:
        score += s["dependency_bonus"]
        tags += deps
    taiwan = bool(name_sets["taiwan"].search(text))
    if taiwan:
        score += s["taiwan_bonus"]
    fund = any(p.search(text) for p in name_sets.get("fund", []))
    return {**item, "categories": hits, "severity": max_sev, "score": score,
            "entities": sorted(set(tags)), "relevant": bool(tags or taiwan or fund)}


def build_name_sets(cfg: dict) -> dict:
    top, other = [], []
    for h in cfg["holdings"]:
        for n in holding_names(h):
            p = compile_names([n])[0]
            (top if h.get("weight", 0) >= 3 else other).append((h["name"], p))
    deps_all = cfg["dependencies"]["customers"] + cfg["dependencies"]["suppliers"]
    deps = [(d, compile_names([d])[0]) for d in deps_all]
    taiwan = re.compile(r"\btaiwan(ese)?\b|\btaipei\b|\bTAIEX\b", re.IGNORECASE)
    fund = compile_names(cfg["fund"]["search_names"])
    return {"top": top, "other": other, "deps": deps, "taiwan": taiwan, "fund": fund}


def split_flagged(classified: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    """Flagged = score >= flag_min_score (and, if require_relevance, names Taiwan/fund/holding/dependency)."""
    s = cfg["scoring"]
    need = s.get("require_relevance", False)
    ok = lambda c: c["score"] >= s["flag_min_score"] and (c.get("relevant", True) or not need)
    return [c for c in classified if ok(c)], [c for c in classified if not ok(c)]


# --------------------------------------------------------------------------- seen-store

def load_seen() -> dict:
    if SEEN_FILE.exists():
        try:
            return json.loads(SEEN_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_seen(seen: dict, retention_days: int, now: dt.datetime) -> None:
    cutoff = (now - dt.timedelta(days=retention_days)).isoformat()
    seen = {k: v for k, v in seen.items() if v >= cutoff}
    SEEN_FILE.write_text(json.dumps(seen, indent=0), encoding="utf-8")


# --------------------------------------------------------------------------- reporting

def cat_label(cat: str) -> str:
    return cat.replace("_", " ").capitalize()


def build_html(flagged, watch, cfg, now_local, n_queries, n_raw) -> str:
    def rows(items):
        out = []
        for i in items:
            cats = "<br>".join(f"<b>{escape(cat_label(c))}</b>: {escape(', '.join(k))}"
                               for c, k in i["categories"].items())
            when = i["ts"].strftime("%d %b %H:%M") if i["ts"] else ""
            ents = escape(", ".join(i["entities"])) or "–"
            out.append(
                f"<tr class=s{i['severity']}><td class=n>{i['score']}</td>"
                f"<td><span class=sev>{SEVERITY_LABEL[i['severity']]}</span></td>"
                f"<td><a href=\"{escape(i['link'])}\">{escape(i['title'])}</a>"
                f"<br><small>{escape(i['source'])} · {when} UTC</small></td>"
                f"<td>{ents}</td><td><small>{cats}</small></td></tr>")
        return "\n".join(out)

    head = "<tr><th>Score</th><th>Severity</th><th>Headline</th><th>Affects</th><th>Why flagged</th></tr>"
    f = cfg["fund"]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fund risk watch {now_local:%Y-%m-%d}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:16px;color:#1d1d1f;max-width:1100px}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} .box{{overflow-x:auto}}
th,td{{border-bottom:1px solid #eee;padding:6px;text-align:left;vertical-align:top}}
td.n{{text-align:right}} small{{color:#666}}
.sev{{font-weight:600;font-size:11px;padding:2px 6px;border-radius:4px}}
tr.s3 .sev{{background:#fde2e1;color:#a4000f}} tr.s2 .sev{{background:#fff1d6;color:#8a5300}}
tr.s1 .sev{{background:#e8eef7;color:#24476b}}
</style></head><body>
<h1>Risk watch · {escape(f['name'])}</h1>
<p><small>ISIN {escape(f['isin'])} · {now_local:%a %d %b %Y, %H:%M} · {n_queries} searches,
{n_raw} articles scanned, new items only. Keyword-based flagging: expect some false positives; read the source.</small></p>
<h2>Flagged · {len(flagged)}</h2>
<div class=box><table>{head}{rows(flagged) or '<tr><td colspan=5>Nothing new above the flag threshold.</td></tr>'}</table></div>
<h2>Watch (lower score) · {len(watch)}</h2>
<div class=box><table>{head}{rows(watch) or '<tr><td colspan=5>None.</td></tr>'}</table></div>
<p><small>Not investment advice.</small></p>
</body></html>"""


def build_markdown(flagged, watch, cfg, now_local) -> str:
    lines = [f"## Risk watch {now_local:%Y-%m-%d} · {cfg['fund']['name']}", "",
             f"**Flagged:** {len(flagged)} · **Watch:** {len(watch)}", ""]
    if flagged:
        lines += ["| Score | Severity | Headline | Affects | Categories |", "|---|---|---|---|---|"]
        for i in flagged[:25]:
            t = i["title"].replace("|", "/")
            lines.append(f"| {i['score']} | {SEVERITY_LABEL[i['severity']]} | [{t}]({i['link']}) "
                         f"| {', '.join(i['entities']) or '–'} | {', '.join(map(cat_label, i['categories']))} |")
    return "\n".join(lines) + "\n"


def append_log(items, now) -> None:
    import csv
    new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run_utc", "published_utc", "score", "severity", "categories",
                        "entities", "source", "title", "link"])
        for i in items:
            w.writerow([now.isoformat(timespec="minutes"),
                        i["ts"].isoformat(timespec="minutes") if i["ts"] else "",
                        i["score"], SEVERITY_LABEL[i["severity"]], ";".join(i["categories"]),
                        ";".join(i["entities"]), i["source"], i["title"], i["link"]])


def send_email(html: str, subject: str, required: bool = False) -> None:
    """required: fail instead of skipping when the SMTP settings are missing (test e-mails)."""
    host, to = os.environ.get("SMTP_HOST"), os.environ.get("EMAIL_TO")
    user, pwd = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    port = int(os.environ.get("SMTP_PORT") or "587")
    if not (host and to and user and pwd):
        missing = [k for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASS", "EMAIL_TO") if not os.environ.get(k)]
        if required:
            sys.exit(f"[error] cannot send test e-mail, not set: {', '.join(missing)}")
        print(f"[info] e-mail skipped, not set: {', '.join(missing)}")
        return
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.set_content("Open in an HTML-capable mail client.")
    msg.add_alternative(html, subtype="html")
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(user, pwd)
        s.send_message(msg)
    print(f"[info] e-mailed to {to}")


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "fund_watch.yaml"))
    ap.add_argument("--no-email", action="store_true")
    ap.add_argument("--all", action="store_true", help="ignore seen-store")
    ap.add_argument("--test-email", action="store_true", help="send the report even if nothing is flagged")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    DATA_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    now = dt.datetime.now(UTC)
    try:
        from zoneinfo import ZoneInfo
        now_local = now.astimezone(ZoneInfo("Europe/Stockholm"))
    except Exception:
        now_local = now

    cats = compile_categories(cfg)
    names = build_name_sets(cfg)
    queries = build_queries(cfg)
    seen = {} if args.all else load_seen()

    raw, unique = 0, {}
    for i, (_, q) in enumerate(queries):
        items = fetch_feed(q, cfg)
        raw += len(items)
        for it in items:
            k = item_key(it["title"])
            if k in seen or k in unique:
                continue
            unique[k] = it
        if i < len(queries) - 1:
            time.sleep(cfg["news"]["pause_seconds"])

    classified = []
    for k, it in unique.items():
        c = classify(it, cats, names, cfg)
        if c:
            classified.append(c)
        seen[k] = now.isoformat()   # mark every new item as seen, flagged or not
    min_score = cfg["scoring"]["flag_min_score"]
    classified.sort(key=lambda x: (x["score"], x["severity"], x["ts"] or now), reverse=True)
    flagged, watch = split_flagged(classified, cfg)

    if not args.all:  # --all is a look-around run: don't touch the seen-store or the log
        save_seen(seen, cfg["news"]["seen_retention_days"], now)
        append_log(flagged + watch, now)

    html = build_html(flagged, watch, cfg, now_local, len(queries), raw)
    (REPORT_DIR / f"fund_watch_{now_local:%Y-%m-%d_%H%M}.html").write_text(html, encoding="utf-8")
    (REPORT_DIR / "fund_watch_latest.html").write_text(html, encoding="utf-8")

    md = build_markdown(flagged, watch, cfg, now_local)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)

    e = cfg.get("email", {})
    high = sum(1 for f in flagged if f["severity"] == 3)
    subject = f"Taiwan fund risk watch {now_local:%d %b}: {len(flagged)} flagged ({high} high)"
    if args.test_email:
        send_email(html, "[TEST] " + subject, required=True)
    elif e.get("enabled") and not args.no_email and (flagged or e.get("send_when") == "always"):
        send_email(html, subject)
    return 0


if __name__ == "__main__":
    sys.exit(main())
