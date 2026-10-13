"""
Company press releases for the LLM experiment -> data/releases.csv.

Primary source: each company's MFN feed, https://mfn.se/all/a/<mfn_slug>.rss?limit=N
(verified 2026-10-07). MFN also carries releases distributed through Cision (their links
contain /cis/), and each item includes the full release text in <x:content type="text">,
so no web pages are fetched. Rows are source_type=primary.

Fallback: Google News items whose publisher matches config release_sources (the wire
names), stored as source_type=aggregator without body text. v0 keeps its own news logic;
v0b and v1 use primary rows only.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from html import unescape
from pathlib import Path

import pandas as pd

try:  # python.org macOS builds ship without CA certificates; use certifi's bundle
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

UTC = dt.UTC
USER_AGENT = "Mozilla/5.0 (stock-attention-scanner)"
MFN_NS = "{https://mfn.se/schemas/rss-ns-x/}"
RELEASE_COLUMNS = [
    "release_id",
    "ticker",
    "source",
    "source_type",
    "published_at",
    "title",
    "url",
    "language",
    "body_text",
    "fetched_at",
]


# --------------------------------------------------------------------------- fetching


def mfn_url(slug: str, limit: int) -> str:
    return f"https://mfn.se/all/a/{urllib.parse.quote(slug)}.rss?limit={limit}"


def fetch_mfn_xml(slug: str, limit: int, timeout: int = 30) -> bytes | None:
    """The raw MFN feed for one company, or None (with a warning) if it can't be fetched."""
    req = urllib.request.Request(mfn_url(slug, limit), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as exc:
        print(f"[warn] MFN feed failed for {slug}: {exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------- parsing


def release_id(source: str, url: str) -> str:
    """Stable id: hash of source + url."""
    return hashlib.sha1(f"{source}|{url}".encode()).hexdigest()[:16]


def clean_text(text: str, max_chars: int) -> str:
    text = unescape(re.sub(r"<[^>]+>", " ", text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _utc_iso(ts: dt.datetime) -> str:
    return ts.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_mfn(
    xml: bytes, ticker: str, fetched_at: dt.datetime, max_chars: int, prefer_language: str = "sv"
) -> list[dict]:
    """Primary release rows from an MFN feed. Swedish and English versions of one release share
    a timestamp; only one is kept (prefer_language when present, otherwise the first)."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        print(f"[warn] MFN feed for {ticker} is not valid XML: {exc}", file=sys.stderr)
        return []
    by_ts = {}
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub = it.findtext("pubDate")
        if not (title and link and pub):
            continue
        try:
            ts = pd.Timestamp(dt.datetime.strptime(pub.strip(), "%a, %d %b %Y %H:%M:%S %z"))
        except ValueError:
            continue
        lang = (it.findtext(f"{MFN_NS}language") or "").strip()
        bodies = {c.get("type"): c.text or "" for c in it.findall(f"{MFN_NS}content")}
        body = bodies.get("text") or bodies.get("html") or it.findtext("description") or ""
        row = {
            "release_id": release_id("mfn", link),
            "ticker": ticker,
            "source": "mfn",
            "source_type": "primary",
            "published_at": _utc_iso(ts.to_pydatetime()),
            "title": title,
            "url": link,
            "language": lang,
            "body_text": clean_text(body, max_chars),
            "fetched_at": _utc_iso(fetched_at),
        }
        key = row["published_at"]
        if key not in by_ts or (lang == prefer_language and by_ts[key]["language"] != prefer_language):
            by_ts[key] = row
    return sorted(by_ts.values(), key=lambda r: r["published_at"], reverse=True)


def aggregator_rows(items: list[dict], ticker: str, release_sources: list[str], fetched_at: dt.datetime) -> list[dict]:
    """Google News items from wire publishers (config release_sources), as aggregator rows."""
    keys = [s.lower() for s in release_sources]
    rows = []
    for i in items:
        src = i.get("source") or ""
        if not i.get("ts") or not any(k in src.lower() for k in keys):
            continue
        rows.append(
            {
                "release_id": release_id(src, i["link"]),
                "ticker": ticker,
                "source": src,
                "source_type": "aggregator",
                "published_at": _utc_iso(i["ts"]),
                "title": i["title"],
                "url": i["link"],
                "language": "",
                "body_text": "",
                "fetched_at": _utc_iso(fetched_at),
            }
        )
    return rows


# --------------------------------------------------------------------------- store


def load_releases(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=RELEASE_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in RELEASE_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[RELEASE_COLUMNS]


def _norm_title(title: str) -> str:
    t = re.sub(r"\s+-\s+[^-]+$", "", title or "").lower()
    return " ".join(re.findall(r"[a-z0-9åäöéü]+", t))


def _near_duplicate(row: dict, others: list[dict], hours: float = 24, ratio: float = 0.9) -> bool:
    """Same ticker, published within `hours`, and (almost) the same title."""
    t0 = pd.Timestamp(row["published_at"])
    n0 = _norm_title(row["title"])
    for o in others:
        if o["ticker"] != row["ticker"]:
            continue
        if abs((pd.Timestamp(o["published_at"]) - t0).total_seconds()) > hours * 3600:
            continue
        n1 = _norm_title(o["title"])
        if n0 == n1 or SequenceMatcher(None, n0, n1).ratio() >= ratio:
            return True
    return False


def merge_releases(existing: pd.DataFrame, new_rows: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Add new rows, skipping known release_ids and near-identical titles for the same ticker
    within 24h. A primary row always wins over an aggregator copy of the same release: primary
    rows are only compared with other primary rows, and an aggregator copy stored earlier (say,
    when the MFN fetch failed) is removed when its primary arrives. Returns (all rows, rows
    actually added)."""
    known_ids = set(existing["release_id"])
    pool = existing.to_dict("records")
    added = []
    for r in sorted(new_rows, key=lambda r: (r["source_type"] != "primary", r["published_at"])):
        if r["release_id"] in known_ids:
            continue
        primary = r["source_type"] == "primary"
        same_ticker = [o for o in pool if o["ticker"] == r["ticker"]]
        rivals = [o for o in same_ticker if o["source_type"] == "primary"] if primary else same_ticker
        if _near_duplicate(r, rivals):
            continue
        if primary:  # an aggregator copy stored earlier gives way to the primary release
            twins = {id(o) for o in same_ticker if o["source_type"] != "primary" and _near_duplicate(o, [r])}
            pool = [o for o in pool if id(o) not in twins]
        known_ids.add(r["release_id"])
        pool.append(r)
        added.append(r)
    out = pd.DataFrame(pool, columns=RELEASE_COLUMNS)
    return out.sort_values(["published_at", "ticker"], kind="stable").reset_index(drop=True), added


def coverage_gaps(existing: pd.DataFrame, feed_rows: list[dict], ticker: str, limit: int) -> str | None:
    """Warning text if the feed may not reach back to the last release already stored: the feed
    returned its full `limit` and even its oldest item is newer than our latest for this ticker."""
    if len(feed_rows) < limit:
        return None
    mine = existing[(existing["ticker"] == ticker) & (existing["source_type"] == "primary")]
    if mine.empty:
        return None
    oldest_in_feed = min(r["published_at"] for r in feed_rows)
    if oldest_in_feed > mine["published_at"].max():
        return (
            f"{ticker}: MFN feed returned {limit} items, all newer than the last stored release; "
            "releases in between may be missing (raise releases.mfn_limit)"
        )
    return None


def save_releases(df: pd.DataFrame, path: Path) -> None:
    df[RELEASE_COLUMNS].to_csv(path, index=False)
