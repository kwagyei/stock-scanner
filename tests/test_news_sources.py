"""Offline tests for news_sources.py (no network)."""

from __future__ import annotations

import datetime as dt

import news_sources as ns
import scanner

UTC = dt.UTC
FETCHED = dt.datetime(2026, 10, 7, 6, 10, tzinfo=UTC)

XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:x="https://mfn.se/schemas/rss-ns-x/"><channel><title>MFN</title>
<item><title>Eltel vinner ramavtal</title><link>https://mfn.se/a/eltel/sv-1</link>
  <pubDate>Tue, 06 Oct 2026 06:00:00 +0000</pubDate><x:language>sv</x:language>
  <x:content type="html">&lt;p&gt;Eltel har &lt;b&gt;vunnit&lt;/b&gt; ett ramavtal.&lt;/p&gt;</x:content>
  <x:content type="text">Eltel har vunnit   ett ramavtal v\xc3\xa4rt 300 MSEK.</x:content></item>
<item><title>Eltel wins frame agreement</title><link>https://mfn.se/a/eltel/en-1</link>
  <pubDate>Tue, 06 Oct 2026 06:00:00 +0000</pubDate><x:language>en</x:language>
  <x:content type="text">Eltel has won a frame agreement.</x:content></item>
<item><title>Half-year report</title><link>https://mfn.se/cis/a/eltel/en-2</link>
  <pubDate>Tue, 21 Jul 2026 06:00:00 +0200</pubDate><x:language>en</x:language>
  <x:content type="html">&lt;p&gt;Sales &amp;amp; profit up&lt;/p&gt;</x:content></item>
</channel></rss>"""


def row(ticker="ELTEL.ST", title="Eltel vinner ramavtal", ts="2026-10-06T06:00:00Z", url=None, source_type="primary"):
    url = url or f"https://mfn.se/a/{title}"
    return {
        "release_id": ns.release_id("mfn", url),
        "ticker": ticker,
        "source": "mfn",
        "source_type": source_type,
        "published_at": ts,
        "title": title,
        "url": url,
        "language": "sv",
        "body_text": "",
        "fetched_at": "2026-10-07T06:10:00Z",
    }


# --------------------------------------------------------------------------- parsing


def test_parse_mfn_keeps_swedish_twin_with_full_text_body():
    rows = ns.parse_mfn(XML, "ELTEL.ST", FETCHED, max_chars=8000)
    assert [r["title"] for r in rows] == ["Eltel vinner ramavtal", "Half-year report"]
    first = rows[0]
    assert first["body_text"] == "Eltel har vunnit ett ramavtal värt 300 MSEK."  # text version, whitespace squashed
    assert first["source_type"] == "primary" and first["language"] == "sv"
    assert first["published_at"] == "2026-10-06T06:00:00Z"
    assert first["fetched_at"] == "2026-10-07T06:10:00Z"
    assert first["release_id"] == ns.release_id("mfn", "https://mfn.se/a/eltel/sv-1")


def test_parse_mfn_html_fallback_converts_to_utc_and_truncates():
    rows = ns.parse_mfn(XML, "ELTEL.ST", FETCHED, max_chars=10)
    half = rows[1]
    assert half["published_at"] == "2026-07-21T04:00:00Z"
    assert half["body_text"] == "Sales & pr"
    assert set(rows[0]) == set(ns.RELEASE_COLUMNS)


def test_parse_mfn_bad_xml_warns(capsys):
    assert ns.parse_mfn(b"<html>oops", "X.ST", FETCHED, 100) == []
    assert "not valid XML" in capsys.readouterr().err


def test_release_id_is_stable_and_source_specific():
    assert ns.release_id("mfn", "u") == ns.release_id("mfn", "u")
    assert ns.release_id("mfn", "u") != ns.release_id("Cision", "u")
    assert len(ns.release_id("mfn", "u")) == 16


def test_aggregator_rows_only_wire_sources():
    ts = dt.datetime(2026, 10, 6, 7, 0, tzinfo=UTC)
    items = [
        {"title": "Eltel wins - Cision News", "link": "g1", "source": "Cision News", "ts": ts},
        {"title": "Eltel stock soars", "link": "g2", "source": "Placera", "ts": ts},
        {"title": "No date", "link": "g3", "source": "MFN", "ts": None},
    ]
    rows = ns.aggregator_rows(items, "ELTEL.ST", ["cision", "mfn"], FETCHED)
    assert [r["url"] for r in rows] == ["g1"]
    assert rows[0]["source_type"] == "aggregator" and rows[0]["body_text"] == ""


# --------------------------------------------------------------------------- store / dedup


def test_merge_skips_known_ids_and_near_identical_titles_within_24h():
    existing = ns.load_releases(ns.Path("/nonexistent/releases.csv"))
    out, added = ns.merge_releases(existing, [row()])
    assert len(added) == 1
    # same id again
    out, added = ns.merge_releases(out, [row()])
    assert added == []
    # aggregator copy with publisher suffix, 3h later -> near duplicate
    agg = row(
        title="Eltel vinner ramavtal - Cision News",
        ts="2026-10-06T09:00:00Z",
        url="https://news.google/x",
        source_type="aggregator",
    )
    out, added = ns.merge_releases(out, [agg])
    assert added == []
    # same title two days later, or another ticker -> kept
    later = row(ts="2026-10-08T09:00:00Z", url="https://mfn.se/a/later")
    other = row(ticker="NCAB.ST", url="https://mfn.se/a/ncab")
    out, added = ns.merge_releases(out, [later, other])
    assert len(added) == 2 and len(out) == 3


def test_merge_prefers_primary_over_aggregator_in_same_batch():
    agg = row(
        title="Eltel vinner ramavtal", url="https://news.google/x", source_type="aggregator", ts="2026-10-06T05:00:00Z"
    )
    out, added = ns.merge_releases(ns.load_releases(ns.Path("/nope")), [agg, row()])
    assert [r["source_type"] for r in added] == ["primary"]


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "releases.csv"
    out, _ = ns.merge_releases(ns.load_releases(path), [row()])
    ns.save_releases(out, path)
    back = ns.load_releases(path)
    assert list(back.columns) == ns.RELEASE_COLUMNS
    assert back.iloc[0]["title"] == "Eltel vinner ramavtal"


def test_coverage_gap_warning_when_feed_is_full_and_newer_than_store():
    stored, _ = ns.merge_releases(ns.load_releases(ns.Path("/nope")), [row(ts="2026-09-01T06:00:00Z")])
    feed = [row(ts=f"2026-10-0{d}T06:00:00Z", url=f"u{d}") for d in (2, 3)]
    assert "may be missing" in ns.coverage_gaps(stored, feed, "ELTEL.ST", limit=2)
    assert ns.coverage_gaps(stored, feed, "ELTEL.ST", limit=3) is None  # feed not full
    assert ns.coverage_gaps(stored, feed, "NCAB.ST", limit=2) is None  # nothing stored
    overlap = feed + [row(ts="2026-09-01T06:00:00Z")]
    assert ns.coverage_gaps(stored, overlap, "ELTEL.ST", limit=3) is None  # reaches back


# --------------------------------------------------------------------------- v0 unchanged


def test_v0_releases_identical_from_shared_bytes(monkeypatch):
    """v0 parses the same bytes news_sources fetched; result must equal the old URL path."""
    cfg = scanner.load_config(scanner.ROOT / "config.yaml")
    real_parse = scanner.feedparser.parse
    monkeypatch.setattr(
        scanner.feedparser, "parse", lambda src, agent=None: real_parse(XML if isinstance(src, str) else src)
    )
    assert scanner.fetch_releases("eltel", cfg, XML) == scanner.fetch_releases("eltel", cfg)
