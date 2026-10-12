"""Offline tests for build_watchlist.py. All network calls are mocked."""

import re
import socket

import pandas as pd
import pytest

import build_watchlist as bw
import scanner


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def guard(*a, **k):
        raise RuntimeError("network access in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(bw.time, "sleep", lambda s: None)
    monkeypatch.setattr(scanner, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(bw, "REVIEW_LOG", tmp_path / "data" / "watchlist_review.csv")


@pytest.mark.parametrize(
    "company,query",
    [
        ("Kambi Group Plc", "Kambi Group"),
        ("ShaMaran Petroleum Ltd", "ShaMaran Petroleum"),
        ("Smart Eye AB (publ)", "Smart Eye"),
        ("Bahnhof", "Bahnhof"),
    ],
)
def test_news_query_strips_legal_form(company, query):
    assert bw.news_query(company) == query


def test_class_suffix_gives_company_stem():
    names = pd.Series(
        ["Episurf B", "Acrinova A", "Implantica SDB", "Navigo Invest Pref", "AAC Clyde Space", "Hilbert Group ser. B"]
    )
    assert list(names.str.replace(bw.CLASS_SUFFIX, "", regex=True)) == [
        "Episurf",
        "Acrinova",
        "Implantica",
        "Navigo Invest",
        "AAC Clyde Space",
        "Hilbert Group",
    ]


def test_build(monkeypatch, tmp_path, capsys):
    wl = tmp_path / "watchlist.csv"
    wl.write_text("ticker,name,query,mfn_slug,title_match\nMAN.ST,Manual Co,Manual,manual-co,Manual\n")
    lists = {
        "SMALL_CAP": [
            ("MAN", "Manual Co"),
            ("BIG A", "Big Co A"),
            ("BIG B", "Big Co B"),
            ("THIN", "Thin Co"),
            ("PRF PREF", "Prf Pref"),
        ],
        "FIRST_NORTH": [("NEW", "New Co"), ("NOMFN", "Nomfn Plc"), ("GONE", "Gone Co")],
    }

    def fake_json(url):
        if "nasdaq" in url:
            seg = "FIRST_NORTH" if "FIRST_NORTH" in url else "SMALL_CAP"
            return {
                "data": {
                    "instrumentListing": {
                        "rows": [
                            {"symbol": s, "fullName": n, "isin": f"SE{s}", "currency": "SEK"} for s, n in lists[seg]
                        ]
                    }
                }
            }
        if "query=Big" in url:
            return [{"slug": "big-co", "isins": ["SEBIG B"], "tickers": []}]
        if "query=Nomfn" in url:
            return [{"slug": "other-co", "isins": ["SEOTHER"], "tickers": []}]
        return []

    turnover = {
        "MAN.ST": (5e5, 60),
        "BIG-A.ST": (2e6, 60),
        "BIG-B.ST": (9e6, 60),
        "THIN.ST": (2e5, 60),
        "NEW.ST": (5e6, 10),
        "NOMFN.ST": (3e6, 60),
    }
    monkeypatch.setattr(bw, "get_json", fake_json)
    monkeypatch.setattr(bw, "median_turnover", lambda tickers, days: {t: turnover[t] for t in tickers if t in turnover})
    monkeypatch.setattr(scanner, "fetch_releases", lambda slug, cfg: [{"title": "x"}])

    assert bw.main(["--watchlist", str(wl)]) == 0

    version = scanner.watchlist_version(wl)
    assert re.fullmatch(r"auto-\d{4}-\d{2}-\d{2}-[0-9a-f]{6}", version)
    out = scanner.load_watchlist(wl)
    assert list(zip(out["ticker"], out["source"], out["mfn_slug"], out["title_match"])) == [
        ("MAN.ST", "manual", "manual-co", "Manual"),  # kept as is, although illiquid
        ("BIG-B.ST", "auto", "big-co", ""),  # most liquid share class only
    ]
    review = pd.read_csv(bw.REVIEW_LOG).set_index("ticker")["reason"].to_dict()
    assert review == {
        "PRF-PREF.ST": "preference share",
        "NEW.ST": "only 10 trading days of history",
        "GONE.ST": "no Yahoo data for symbol",
        "NOMFN.ST": "no MFN company found for ISIN",
    }

    # A second build keeps manual rows and reuses known slugs; only failures are retried.
    looked_up = []
    monkeypatch.setattr(bw, "mfn_slug", lambda company, isin, symbol: looked_up.append(symbol))
    assert bw.main(["--watchlist", str(wl)]) == 0
    assert looked_up == ["NOMFN"]
    assert list(scanner.load_watchlist(wl)["ticker"]) == ["MAN.ST", "BIG-B.ST"]
    assert scanner.watchlist_version(wl) == version  # same members, same version

    turnover["BIG-B.ST"] = (1e5, 60)  # now illiquid -> membership changes
    assert bw.main(["--watchlist", str(wl)]) == 0
    assert scanner.watchlist_version(wl) != version
