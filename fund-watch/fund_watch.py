#!/usr/bin/env python3
"""
Fund risk news monitor.

Scans Google News daily for anything that could hurt the funds in fund_watch.yaml: their
holdings, the companies those depend on, geopolitics, trade restrictions, natural disasters,
infrastructure, labour, legal issues, macro and market stress. Flags and ranks the issues
per fund. Articles are fetched once and scored for every fund, so one article can be
flagged for several funds.

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
import csv
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
SUMMARY_FILE = DATA_DIR / "fund_watch_latest.json"   # read by the stock scanner's dashboard
LOG_COLUMNS = ["run_utc", "fund", "published_utc", "score", "severity", "categories",
               "entities", "source", "title", "link"]
UTC = dt.timezone.utc
SEVERITY_LABEL = {3: "HIGH", 2: "MEDIUM", 1: "LOW"}


# --------------------------------------------------------------------------- config / queries

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def holding_names(h: dict) -> list[str]:
    return [h["name"], *h.get("aliases", [])]


def build_queries(cfg: dict) -> list[tuple[str, str]]:
    """(group, query) pairs for all funds, without duplicates."""
    return list(dict.fromkeys(gq for f in cfg["funds"] for gq in fund_queries(f)))


def query_funds(cfg: dict) -> dict:
    """query -> ids of the funds that search for it."""
    out = {}
    for f in cfg["funds"]:
        for _, q in fund_queries(f):
            out.setdefault(q, set()).add(f["id"])
    return out


def fund_queries(f: dict) -> list[tuple[str, str]]:
    """(group, query) pairs for one fund."""
    q = []
    for name in f["search_names"]:
        q.append(("fund", f'"{name}"'))
    for h in f["holdings"]:
        q.append(("holding", " OR ".join(f'"{n}"' for n in holding_names(h))))
    deps = f.get("dependencies") or {}
    for group in ("customers", "suppliers"):
        context = deps.get(f"search_{group}")
        if context:   # dependencies without search words are not searched separately
            q += [(group[:-1], f'"{c}" {context}') for c in deps.get(group, [])]
    q += [("topic", t) for t in f.get("topics", [])]
    return q


def compile_categories(cfg: dict) -> dict:
    """{category: (severity, keywords, keyword patterns, exclude patterns)}.

    exclude = regular expressions; text matching one is ignored for that category
    (e.g. "strike ... deal" for labour).
    """
    out = {}
    for cat, spec in cfg["risk_categories"].items():
        pats = [re.compile(r"(?<![\w-])" + re.escape(k) + r"(?![\w-])", re.IGNORECASE)
                for k in spec["keywords"]]
        excl = [re.compile(x, re.IGNORECASE) for x in spec.get("exclude", [])]
        out[cat] = (spec["severity"], spec["keywords"], pats, excl)
    return out


def compile_names(names: list[str]) -> list[re.Pattern]:
    return [re.compile(r"(?<!\w)" + re.escape(n) + r"(?!\w)", re.IGNORECASE) for n in names]


class _Name:
    """A company name pattern that ignores text matching its exclude phrases ("apple crop")."""

    def __init__(self, name: str, exclude=()):
        self.pattern = compile_names([name])[0]
        self.exclude = [re.compile(x, re.IGNORECASE) for x in exclude]

    def search(self, text: str):
        for x in self.exclude:
            text = x.sub(" ", text)
        return self.pattern.search(text)


def build_name_sets(fund: dict) -> dict:
    """Patterns for one fund: top / other holdings, dependencies, relevance terms, fund names."""
    top, other = [], []
    cut = fund.get("top_holding_min_weight", 3)
    for h in fund["holdings"]:
        for n in holding_names(h):
            (top if h.get("weight", 0) >= cut else other).append((h["name"], _Name(n, h.get("exclude", []))))
    deps = fund.get("dependencies") or {}
    dep_names = deps.get("customers", []) + deps.get("suppliers", [])
    # Dependencies the fund doesn't search for itself: it relies on other funds' searches for them.
    unsearched = [d for g in ("customers", "suppliers") if not deps.get(f"search_{g}") for d in deps.get(g, [])]
    return {"top": top, "other": other,
            "deps": [(d, _Name(d)) for d in dep_names],
            "deps_unsearched": [(d, _Name(d)) for d in unsearched],
            "region": compile_names(fund.get("relevance_terms", [])),
            "fund": compile_names(fund["search_names"])}


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

def article_text(item: dict) -> str:
    """Headline + summary without the publisher's name, which Google News adds to both
    ("... - Taipei Times", "Simply Wall Street"): it says nothing about the story."""
    text = f"{item['title']} {item['summary']}"
    src = (item.get("source") or "").strip()
    return text.replace(src, " ") if src else text


def match_categories(item: dict, cats: dict) -> dict:
    """Risk keywords found in the article: {category: [keywords]}. Fund-independent."""
    text = article_text(item)
    hits = {}
    for cat, (sev, kws, pats, excl) in cats.items():
        t = text
        for x in excl:
            t = x.sub(" ", t)
        matched = [kw for kw, p in zip(kws, pats) if p.search(t)]
        if matched:
            hits[cat] = matched
    return hits


def classify(item: dict, cats: dict, name_sets: dict, cfg: dict, matched=None, fund: dict | None = None) -> dict | None:
    """Score an article for one fund (name_sets from build_name_sets). None if no risk keyword.

    fund: its `severity` overrides per category apply (e.g. legal news is LOW for an index fund).
    """
    hits = matched if matched is not None else match_categories(item, cats)
    if not hits:
        return None
    overrides = (fund or {}).get("severity") or {}
    max_sev = max(overrides.get(c, cats[c][0]) for c in hits)
    text = article_text(item)
    s = cfg["scoring"]
    score, tags = max_sev, []
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
    region = any(p.search(text) for p in name_sets["region"])
    if region:
        score += s["region_bonus"]
    named = any(p.search(text) for p in name_sets["fund"])
    return {**item, "categories": hits, "severity": max_sev, "score": score,
            "entities": sorted(set(tags)), "relevant": bool(tags or region or named)}


def split_flagged(classified: list[dict], cfg: dict, fund: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Flagged = score >= flag_min_score (the fund's own, if set) and, if require_relevance, tied to the fund."""
    s = cfg["scoring"]
    need = s.get("require_relevance", False)
    min_score = (fund or {}).get("flag_min_score", s["flag_min_score"])
    ok = lambda c: c["score"] >= min_score and (c.get("relevant", True) or not need)
    return [c for c in classified if ok(c)], [c for c in classified if not ok(c)]


def _considered(item: dict, fund: dict, names: dict) -> bool:
    """A fund looks at articles from its own searches, plus articles from other funds' searches
    that name one of its holdings, relevance words, or dependencies it doesn't search for itself
    (the global fund's TSMC). A dependency the fund does search for (with its own context words)
    doesn't pull in other funds' broad news, e.g. all Google news into the Taiwan fund.
    item["funds"] = ids of the funds whose searches found the article."""
    if fund["id"] in item.get("funds", (fund["id"],)):
        return True
    text = article_text(item)
    return any(p.search(text) for _, p in names["top"] + names["other"] + names["deps_unsearched"]) or \
        any(p.search(text) for p in names["region"] + names["fund"])


def classify_funds(items: list[dict], cfg: dict, now: dt.datetime) -> list[dict]:
    """Per fund: {"fund", "flagged", "watch"}, each list ranked by score."""
    cats = compile_categories(cfg)
    matched = [(it, match_categories(it, cats)) for it in items]
    results = []
    for f in cfg["funds"]:
        names = build_name_sets(f)
        cl = [c for it, m in matched
              if m and _considered(it, f, names) and (c := classify(it, cats, names, cfg, m, f))]
        cl.sort(key=lambda x: (x["score"], x["severity"], x["ts"] or now), reverse=True)
        flagged, watch = split_flagged(cl, cfg, f)
        results.append({"fund": f, "flagged": flagged, "watch": watch})
    return results


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


def _n_high(items) -> int:
    return sum(1 for i in items if i["severity"] == 3)


def build_html(results, now_local, n_queries, n_raw) -> str:
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
    sections = "\n".join(
        f"""<h2>{escape(r['fund']['icon'])} {escape(r['fund']['name'])} <small>ISIN {escape(r['fund']['isin'])}</small></h2>
<h3>Flagged · {len(r['flagged'])}</h3>
<div class=box><table>{head}{rows(r['flagged']) or '<tr><td colspan=5>Nothing new above the flag threshold.</td></tr>'}</table></div>
<h3>Watch (lower score) · {len(r['watch'])}</h3>
<div class=box><table>{head}{rows(r['watch']) or '<tr><td colspan=5>None.</td></tr>'}</table></div>"""
        for r in results)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fund risk watch {now_local:%Y-%m-%d}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:16px;color:#1d1d1f;max-width:1100px}}
h1{{font-size:20px}} h2{{font-size:17px;margin-top:32px;border-bottom:2px solid #ddd;padding-bottom:4px}}
h3{{font-size:14px;margin-top:18px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} .box{{overflow-x:auto}}
th,td{{border-bottom:1px solid #eee;padding:6px;text-align:left;vertical-align:top}}
td.n{{text-align:right}} small{{color:#666}}
.sev{{font-weight:600;font-size:11px;padding:2px 6px;border-radius:4px}}
tr.s3 .sev{{background:#fde2e1;color:#a4000f}} tr.s2 .sev{{background:#fff1d6;color:#8a5300}}
tr.s1 .sev{{background:#e8eef7;color:#24476b}}
</style></head><body>
<h1>Fund risk watch · {now_local:%a %d %b %Y, %H:%M}</h1>
<p><small>{n_queries} searches, {n_raw} articles scanned, new items only. Keyword-based flagging: expect some
false positives; read the source.</small></p>
{sections}
<p><small>Not investment advice.</small></p>
</body></html>"""


SEVERITY_ICON = {3: "🔴 HIGH", 2: "🟠 MEDIUM", 1: "🔵 LOW"}


def _md(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _md_items(items, empty: str) -> str:
    if not items:
        return f"_{empty}_\n"
    out = ["| Score | Severity | Headline | Affects | Why flagged |", "|:---:|:---|:---|:---|:---|"]
    for i in items:
        when = i["ts"].strftime("%d %b %H:%M") if i["ts"] else ""
        why = "<br>".join(f"**{cat_label(c)}:** {_md(', '.join(k))}" for c, k in i["categories"].items())
        out.append(f"| **{i['score']}** | {SEVERITY_ICON[i['severity']]} | [{_md(i['title'])}](<{i['link']}>)"
                   f"<br><sub>{_md(i['source'])} · {when} UTC</sub> | {_md(', '.join(i['entities'])) or '–'} | {why} |")
    return "\n".join(out) + "\n"


def build_page_md(results, now_local, n_queries, n_raw) -> str:
    """The report as a Markdown page (GitHub shows Markdown, not HTML, in a private repo)."""
    back = ("[![Dashboard](https://img.shields.io/badge/Dashboard-24292f?style=for-the-badge)]"
            "(../../DASHBOARD.md)")
    glance = (["| Fund | 🚩 Flagged | 🔴 High | 👀 Watch |", "|:---|:---:|:---:|:---:|"]
              + [f"| {r['fund']['icon']} **{_md(r['fund']['short'])}** | **{len(r['flagged'])}** | "
                 f"**{_n_high(r['flagged'])}** | {len(r['watch'])} |" for r in results])
    out = ["# 🛡️ Fund risk watch", "", back, "",
           f"<sub>Updated **{now_local:%a %d %b %Y, %H:%M}** (Stockholm) · {n_raw} articles from "
           f"{n_queries} searches · new articles only · each article is scored for every fund</sub>", "",
           "\n".join(glance), "",
           "<sub>Score = severity (HIGH 3, MEDIUM 2, LOW 1) + 2 for a top holding (+1 another holding) "
           "+ 1 dependency + 1 relevance word (e.g. Taiwan, Wall Street, krona). Flagged = score 3 or more "
           "and tied to the fund (a relevance word, the fund, a holding or a dependency). Keyword-based: "
           "read the source before acting.</sub>", ""]
    for r in results:
        f, high = r["fund"], _n_high(r["flagged"])
        out += [f"## {f['icon']} {_md(f['name'])}", "", f"<sub>ISIN {f['isin']}</sub>", ""]
        if high:
            out += [f"> [!WARNING]\n> **{high} high-severity item(s)** among the new articles. Read the sources.", ""]
        out += [_md_items(r["flagged"], "✅ Nothing new above the flag threshold."),
                f"<details><summary>👀 Watch: lower score or not clearly linked to the fund "
                f"({len(r['watch'])})</summary>", "", _md_items(r["watch"], "None."), "</details>", ""]
    return "\n".join(out + ["---", "<sub>Research and alerting only. Not investment advice.</sub>", ""])


def write_summary(results, now, now_local, n_raw) -> None:
    """Small JSON summary of the latest run for the stock scanner's dashboard."""
    funds = [{"id": r["fund"]["id"], "short": r["fund"]["short"], "icon": r["fund"]["icon"],
              "name": r["fund"]["name"], "isin": r["fund"]["isin"],
              "n_flagged": len(r["flagged"]), "n_high": _n_high(r["flagged"]), "n_watch": len(r["watch"]),
              "top": [{"score": i["score"], "severity": SEVERITY_LABEL[i["severity"]], "title": i["title"],
                       "link": i["link"], "source": i["source"], "entities": i["entities"],
                       "categories": list(i["categories"])} for i in r["flagged"][:10]]}
             for r in results]
    SUMMARY_FILE.write_text(json.dumps({
        "run_utc": now.isoformat(timespec="minutes"), "run_local": f"{now_local:%Y-%m-%d %H:%M}",
        "n_raw": n_raw, "funds": funds}, indent=1, ensure_ascii=False), encoding="utf-8")


def build_markdown(results, now_local) -> str:
    """Short version for the GitHub job summary."""
    lines = [f"## Fund risk watch {now_local:%Y-%m-%d}", ""]
    for r in results:
        f = r["fund"]
        lines += [f"### {f['icon']} {f['name']}", "",
                  f"**Flagged:** {len(r['flagged'])} · **Watch:** {len(r['watch'])}", ""]
        if r["flagged"]:
            lines += ["| Score | Severity | Headline | Affects | Categories |", "|---|---|---|---|---|"]
            for i in r["flagged"][:25]:
                t = i["title"].replace("|", "/")
                lines.append(f"| {i['score']} | {SEVERITY_LABEL[i['severity']]} | [{t}]({i['link']}) "
                             f"| {', '.join(i['entities']) or '–'} | {', '.join(map(cat_label, i['categories']))} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def append_log(results, now) -> None:
    """One row per (fund, item). Older single-fund logs get a fund column (the Taiwan fund)."""
    if LOG_FILE.exists():
        with open(LOG_FILE, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if rows and "fund" not in rows[0]:
            old_head = rows[0]
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(LOG_COLUMNS)
                for r in rows[1:]:
                    rec = dict(zip(old_head, r), fund="jpm_taiwan")
                    w.writerow([rec.get(c, "") for c in LOG_COLUMNS])
    new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(LOG_COLUMNS)
        for r in results:
            for i in r["flagged"] + r["watch"]:
                w.writerow([now.isoformat(timespec="minutes"), r["fund"]["id"],
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


def email_subject(results, now_local) -> str:
    parts = [f"{r['fund']['short']} {len(r['flagged'])} flagged ({_n_high(r['flagged'])} high)" for r in results]
    return f"Fund risk watch {now_local:%d %b}: " + " · ".join(parts)


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

    queries = build_queries(cfg)
    owners = query_funds(cfg)
    seen = {} if args.all else load_seen()

    raw, unique = 0, {}
    for i, (_, q) in enumerate(queries):
        items = fetch_feed(q, cfg)
        raw += len(items)
        for it in items:
            k = item_key(it["title"])
            if k in seen:
                continue
            if k in unique:     # same story from another search: remember which funds found it
                unique[k]["funds"] |= owners[q]
                continue
            unique[k] = {**it, "funds": set(owners[q])}
        if i < len(queries) - 1:
            time.sleep(cfg["news"]["pause_seconds"])

    results = classify_funds(list(unique.values()), cfg, now)
    for k in unique:
        seen[k] = now.isoformat()   # mark every new item as seen, flagged or not

    if not args.all:  # --all is a look-around run: don't touch the seen-store, log or summary
        save_seen(seen, cfg["news"]["seen_retention_days"], now)
        append_log(results, now)
        write_summary(results, now, now_local, raw)

    html = build_html(results, now_local, len(queries), raw)
    (REPORT_DIR / f"fund_watch_{now_local:%Y-%m-%d_%H%M}.html").write_text(html, encoding="utf-8")
    (REPORT_DIR / "fund_watch_latest.html").write_text(html, encoding="utf-8")
    (REPORT_DIR / "fund_watch_latest.md").write_text(build_page_md(results, now_local, len(queries), raw),
                                                    encoding="utf-8")

    md = build_markdown(results, now_local)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md)
    print(md)

    e = cfg.get("email", {})
    any_flagged = any(r["flagged"] for r in results)
    subject = email_subject(results, now_local)
    if args.test_email:
        send_email(html, "[TEST] " + subject, required=True)
    elif e.get("enabled") and not args.no_email and (any_flagged or e.get("send_when") == "always"):
        send_email(html, subject)
    return 0


if __name__ == "__main__":
    sys.exit(main())
