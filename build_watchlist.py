#!/usr/bin/env python3
"""
Build watchlist.csv from Nasdaq Stockholm Small Cap + First North Stockholm.

Run by hand (e.g. monthly), review the output, then commit:
  python build_watchlist.py              # writes watchlist.csv + data/watchlist_review.csv
  python build_watchlist.py --dry-run    # report only, write nothing

Steps:
  1. Instrument lists from Nasdaq Nordic's share screener (2 requests).
  2. Yahoo symbol per instrument (SYMBOL with spaces -> dashes, + ".ST").
  3. Liquidity filter: median daily close x volume over the last N days (Yahoo, batched).
     One share class per company (the most liquid); preference shares are skipped.
  4. MFN slug matched by ISIN via MFN's company search, then checked to return releases.
  5. Rows marked source=manual in the current watchlist are kept unchanged. Everything
     that fails a check goes to data/watchlist_review.csv instead of the watchlist.

Changing the watchlist changes what the paper-trade test measures, so each build stamps a
new "# version:" line; new paper trades are tagged with it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf

import scanner

NASDAQ_URL = ("https://api.nasdaq.com/api/nordic/screener/shares"
              "?category={category}&tableonly=true&market=STO{segment}")
MFN_SEARCH = "https://mfn.se/search/companies?query={q}"
CLASS_SUFFIX = re.compile(r"\s+(ser\.?\s*)?(A|B|C|D|SDB|A SDB|STAM|PREF|Pref)$")
LEGAL_SUFFIX = re.compile(r"\s+(AB|\(publ\)|Plc|plc|PLC|Ltd\.?|Inc\.?|Corp\.?|ASA|A/S|Oyj|SE|N\.V\.)$")
REVIEW_LOG = scanner.DATA_DIR / "watchlist_review.csv"
WATCHLIST_COLUMNS = ["ticker", "name", "query", "mfn_slug", "source", "title_match"]
PAUSE = 1.0


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": scanner.USER_AGENT,
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def fetch_universe(segments: list[str]) -> pd.DataFrame:
    rows = []
    for seg in segments:
        if seg == "FIRST_NORTH":
            url = NASDAQ_URL.format(category="FIRST_NORTH", segment="")
        else:
            url = NASDAQ_URL.format(category="MAIN_MARKET", segment=f"&segment={seg}")
        data = (get_json(url) or {}).get("data")
        if not data:
            sys.exit(f"[error] Nasdaq returned no data for {seg}: {url}")
        for r in data["instrumentListing"]["rows"]:
            rows.append({"symbol": r["symbol"], "full_name": r["fullName"], "isin": r["isin"],
                         "currency": r["currency"], "list": seg})
        time.sleep(PAUSE * 2)
    df = pd.DataFrame(rows).drop_duplicates("isin")
    df["ticker"] = df["symbol"].str.replace(" ", "-") + ".ST"
    df["company"] = df["full_name"].str.replace(CLASS_SUFFIX, "", regex=True).str.strip()
    return df


def median_turnover(tickers: list[str], days: int, batch: int = 50) -> dict[str, tuple[float, int]]:
    """ticker -> (median close x volume over the last `days` bars, bars available)."""
    out = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        data = yf.download(chunk, period="6mo", interval="1d", group_by="ticker",
                           auto_adjust=False, threads=False, progress=False)
        for t in chunk:
            try:
                h = data[t].dropna(subset=["Close"]).tail(days)
            except KeyError:
                continue
            if len(h):
                out[t] = (float((h["Close"] * h["Volume"]).median()), len(h))
        time.sleep(PAUSE * 2)
    return out


def mfn_slug(company: str, isin: str, symbol: str) -> str | None:
    """MFN slug for the entity carrying this ISIN (or Stockholm ticker); None if not found."""
    for q in dict.fromkeys([company, company.split()[0]]):
        try:
            results = get_json(MFN_SEARCH.format(q=urllib.parse.quote(q))) or []
        except Exception as exc:
            print(f"[warn] MFN search failed for {q}: {exc}", file=sys.stderr)
            results = []
        time.sleep(PAUSE)
        for r in results:
            if isin in (r.get("isins") or []) or f"XSTO:{symbol}" in (r.get("tickers") or []):
                return r["slug"]
    return None


def news_query(company: str) -> str:
    """Exact-phrase news query: the company name without legal-form suffixes."""
    q = company
    while LEGAL_SUFFIX.search(q):
        q = LEGAL_SUFFIX.sub("", q).strip()
    return q


def read_current(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)
    wl = scanner.load_watchlist(path)
    if "source" not in wl.columns:
        wl["source"] = "manual"           # pre-builder watchlist: everything was hand-picked
    wl["source"] = wl["source"].replace("", "manual")
    return wl


def write_watchlist(path: Path, rows: pd.DataFrame, version: str, n_auto: int) -> None:
    header = [
        "# ticker = Yahoo Finance symbol (Stockholm listings end in .ST).",
        "# query  = news search phrase. mfn_slug = company slug on mfn.se (blank = Google News only).",
        "# title_match = optional word a news headline must contain (whole word, exact case);",
        "#          for names that are also common words, where Google News matches loosely.",
        "# source = manual rows are kept by build_watchlist.py; auto rows are rebuilt on every run.",
        "#          To customise an auto row (e.g. a better query), change its source to manual.",
        f"# version: {version}",
        f"# built {dt.date.today().isoformat()} by build_watchlist.py: {n_auto} auto rows "
        "(Nasdaq Stockholm Small Cap + First North, liquidity-filtered)",
    ]
    cols = WATCHLIST_COLUMNS
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(header) + "\n")
        rows[cols].to_csv(fh, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(scanner.ROOT / "config.yaml"))
    ap.add_argument("--watchlist", default=str(scanner.ROOT / "watchlist.csv"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = scanner.load_config(Path(args.config))
    uc = cfg["universe"]
    wl_path = Path(args.watchlist)
    current = read_current(wl_path)
    manual = current[current["source"] == "manual"]
    known_slugs = dict(zip(current["ticker"], current["mfn_slug"]))

    uni = fetch_universe(uc["segments"])
    print(f"[info] universe: {len(uni)} instruments "
          + ", ".join(f"{k} {v}" for k, v in uni["list"].value_counts().items()))
    review = []

    def flag(r, reason, turnover=None):
        review.append({"ticker": r["ticker"], "name": r["full_name"], "list": r["list"],
                       "reason": reason, "median_turnover_sek":
                       "" if turnover is None else round(turnover)})

    skip = uni["symbol"].str.endswith(" PREF") | (uni["currency"] != "SEK")
    for _, r in uni[skip].iterrows():
        flag(r, "preference share" if r["symbol"].endswith(" PREF") else f"currency {r['currency']}")
    uni = uni[~skip]

    print(f"[info] fetching {uc['turnover_days']}-day turnover for {len(uni)} tickers …")
    turnover = median_turnover(list(uni["ticker"]), uc["turnover_days"])
    uni = uni.assign(turnover=uni["ticker"].map(lambda t: turnover.get(t, (None, 0))[0]),
                     bars=uni["ticker"].map(lambda t: turnover.get(t, (None, 0))[1]))

    liquid, n_illiquid = [], 0
    for _, r in uni.iterrows():
        if r["turnover"] is None or pd.isna(r["turnover"]):
            flag(r, "no Yahoo data for symbol")
        elif r["bars"] < uc["min_history_days"]:
            flag(r, f"only {r['bars']} trading days of history", r["turnover"])
        elif r["turnover"] < uc["min_median_turnover_sek"]:
            n_illiquid += 1  # expected for most First North names, not worth reviewing
        else:
            liquid.append(r)
    liquid = pd.DataFrame(liquid)
    # One share class per company: keep the most liquid.
    liquid = liquid.sort_values("turnover", ascending=False).drop_duplicates("company")
    liquid = liquid[~liquid["ticker"].isin(manual["ticker"])]
    print(f"[info] {len(liquid)} liquid companies not already manual; matching MFN slugs …")

    auto = []
    for _, r in liquid.sort_values("company").iterrows():
        slug = known_slugs.get(r["ticker"]) or mfn_slug(r["company"], r["isin"], r["symbol"])
        if not slug:
            flag(r, "no MFN company found for ISIN", r["turnover"])
            continue
        if known_slugs.get(r["ticker"]) != slug:
            rel = scanner.fetch_releases(slug, cfg)
            time.sleep(PAUSE)
            if not rel:
                flag(r, f"MFN feed '{slug}' returned no releases", r["turnover"])
                continue
        auto.append({"ticker": r["ticker"], "name": r["company"], "query": news_query(r["company"]),
                     "mfn_slug": slug, "source": "auto"})
    auto = pd.DataFrame(auto, columns=WATCHLIST_COLUMNS)

    old_auto = set(current.loc[current["source"] == "auto", "ticker"])
    added, removed = sorted(set(auto["ticker"]) - old_auto), sorted(old_auto - set(auto["ticker"]))
    # Date plus a hash of the membership: any change in tickers gives a new version.
    members = ",".join(sorted(set(manual["ticker"]) | set(auto["ticker"])))
    version = f"auto-{dt.date.today().isoformat()}-{hashlib.sha1(members.encode()).hexdigest()[:6]}"
    print(f"[info] manual rows kept: {len(manual)} · auto rows: {len(auto)} "
          f"(+{len(added)} / -{len(removed)} vs current) · below liquidity floor: {n_illiquid} "
          f"· for review: {len(review)}")
    if removed:
        print(f"[info] dropped auto rows: {', '.join(removed)}")

    if args.dry_run:
        print("[info] dry run: nothing written")
        return 0
    rows = pd.concat([manual, auto], ignore_index=True).fillna("")
    write_watchlist(wl_path, rows, version, len(auto))
    scanner.DATA_DIR.mkdir(exist_ok=True)
    pd.DataFrame(review, columns=["ticker", "name", "list", "reason", "median_turnover_sek"]) \
        .to_csv(REVIEW_LOG, index=False)
    print(f"[info] wrote {wl_path.name} ({len(rows)} rows, version {version}) and {REVIEW_LOG.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
