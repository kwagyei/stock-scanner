"""
The three paper strategies of the LLM experiment, and the release timing they share.

  v0                 the original rules (scanner.score_breakdown), unchanged
  v0b                v0, but "company release" counts only primary (MFN/Cision) releases
                     that fall on the signal bar (see below)
  v1_<prompt>        v0b with the flat +2 replaced by LLM points (llm.points), and no buy
                     when a material release scores direction <= llm.block_if_direction_lte

Entry timing (v0b, v1 and the event study): a release published before 09:00 Stockholm on a
trading day is tradable at that day's open; otherwise at the next trading day's open.
Trading days are the dates in the price data. The signal bar is the trading day before
the entry day, so a release belongs to bar D when it was published from 09:00 on D up
to 09:00 on the next trading day. Paper trades enter at the first open after the signal
bar, as v0 already does.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Collection, Iterable
from zoneinfo import ZoneInfo

import pandas as pd

STOCKHOLM = ZoneInfo("Europe/Stockholm")
DateLike = str | dt.date | dt.datetime | pd.Timestamp  # a date or moment in any form the callers use
OPEN = dt.time(9, 0)
V0, V0B = "v0", "v0b"
RELEASE_POINTS_V0 = 2


def v1_label(cfg: dict) -> str:
    return f"v1_{cfg['llm']['prompt_version']}"


def llm_enabled(cfg: dict) -> bool:
    """llm.enabled (default true). Off: no API calls at all, and no v1 book; v0 and v0b still run."""
    return bool(cfg.get("llm")) and bool(cfg["llm"].get("enabled", True))


def strategy_labels(cfg: dict) -> list[str]:
    return [V0, V0B, v1_label(cfg)] if llm_enabled(cfg) else [V0, V0B]


def entry_allowed(row: dict, filters: dict, book: pd.DataFrame) -> bool:
    """Entry filters for one signal (row from the scanner), given the strategy's own paper book.
    No filters = everything allowed.
      max_signal_move: skip when |1-day move on the signal day| is larger.
      cooldown_days_after_stop: skip when the book was stopped out of this stock fewer than this
        many weekdays before the signal day."""
    limit = filters.get("max_signal_move")
    if limit is not None and row.get("chg_1d") is not None and abs(row["chg_1d"]) > limit:
        return False
    days = filters.get("cooldown_days_after_stop")
    if days and not book.empty:
        stops = book[(book["ticker"] == row["ticker"]) & (book["exit_reason"] == "stop") & (book["exit_date"] != "")]
        signal = pd.Timestamp(row["date"])
        for exit_day in pd.to_datetime(stops["exit_date"]):
            if exit_day <= signal and len(pd.bdate_range(exit_day, signal)) - 1 < days:
                return False
    return True


# --------------------------------------------------------------------------- timing


def _local(ts: DateLike) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(STOCKHOLM)


def open_at(day: DateLike) -> pd.Timestamp:
    """09:00 Stockholm on `day`, as an aware timestamp."""
    d = pd.Timestamp(day).date()
    return pd.Timestamp(dt.datetime.combine(d, OPEN), tz=STOCKHOLM)


def entry_day_for(published_at: DateLike, trading_days: Iterable) -> pd.Timestamp | None:
    """First trading day whose 09:00 open is after the release (None if not in the data yet)."""
    t = _local(published_at)
    for d in sorted(pd.DatetimeIndex(trading_days)):
        if open_at(d) > t:
            return pd.Timestamp(d).normalize()
    return None


def signal_bar_for(published_at: DateLike, trading_days: Iterable) -> pd.Timestamp | None:
    """The trading day before the entry day (None if the entry day is unknown or is the first day)."""
    days = sorted(pd.DatetimeIndex(trading_days).normalize())
    entry = entry_day_for(published_at, days)
    if entry is None:
        return None
    i = days.index(entry)
    return days[i - 1] if i > 0 else None


def as_of_for_bar(
    bar_date: DateLike, now: dt.datetime, market_days: Collection, cutoff: dt.time = dt.time(18, 0)
) -> pd.Timestamp:
    """The moment a signal on `bar_date` becomes tradable: 09:00 Stockholm on the next trading
    day, or `now` if that is earlier. News and releases count for the bar only if published
    before it, so a run during the next session (or a late run still on an older bar) can't use
    news that came out after the open the trade buys at.

    The next trading day is the first weekday after the bar that has a bar in `market_days`
    (dates in the price data) or hasn't finished yet (its bar isn't complete before `cutoff`).
    A weekday that is over without any bar was a holiday and is skipped. With no market data at
    all, every weekday counts as a trading day."""
    now_l = _local(now)
    days = set(pd.DatetimeIndex(list(market_days)).normalize()) if len(market_days) else set()
    d = pd.Timestamp(bar_date).normalize()
    for _ in range(10):  # never more than a few holidays in a row
        d = d + pd.offsets.BDay(1)
        over = now_l.date() > d.date() or (now_l.date() == d.date() and now_l.time() >= cutoff)
        if not days or d in days or not over:
            break
    return min(now_l, open_at(d))


def market_trading_days(series_days: Iterable[Iterable], benchmark_days: Iterable = ()) -> set:
    """Dates the market traded, judged from the price data: a date the benchmark has, or one that
    at least two series have (a row counts even if its close is missing). One stray row can't
    turn a holiday into a trading day."""
    from collections import Counter

    counts = Counter(pd.Timestamp(d).normalize() for days in series_days for d in set(days))
    bench = {pd.Timestamp(d).normalize() for d in benchmark_days}
    return bench | {d for d, n in counts.items() if n >= 2}


def inferred_holidays(
    market_days: Iterable, now: dt.datetime, cutoff: dt.time = dt.time(18, 0), lookback_days: int = 7
) -> list[pd.Timestamp]:
    """Weekdays in the last `lookback_days` that are over but have no prices anywhere. as_of_for_bar
    treats them as holidays; if one wasn't, the data for it is missing and the bar before it is
    scored with that day's news (see the limitation in docs/OPERATIONS.md)."""
    now_l = _local(now)
    days = {pd.Timestamp(d).normalize() for d in market_days}
    if not days:
        return []
    out = []
    for d in pd.bdate_range(now_l.date() - dt.timedelta(days=lookback_days), now_l.date()):
        over = now_l.date() > d.date() or (now_l.date() == d.date() and now_l.time() >= cutoff)
        if over and d.normalize() not in days and d.normalize() > min(days):
            out.append(d.normalize())
    return out


def releases_for_bar(releases: pd.DataFrame, ticker: str, bar_date: DateLike, as_of: DateLike) -> pd.DataFrame:
    """Primary releases for `ticker` that belong to signal bar `bar_date`: published from 09:00
    on that day up to `as_of` (as_of_for_bar: the next open, or now if earlier)."""
    if releases.empty:
        return releases
    lo, hi = open_at(bar_date), _local(as_of)
    r = releases[(releases["ticker"] == ticker) & (releases["source_type"] == "primary")]
    pub = pd.to_datetime(r["published_at"], utc=True).dt.tz_convert(STOCKHOLM)
    return r[(pub >= lo) & (pub < hi)]


# --------------------------------------------------------------------------- points


def is_routine(title: str, cfg: dict) -> bool:
    t = (title or "").lower()
    return any(k.lower() in t for k in cfg.get("release_ignore_titles") or [])


def llm_points(score: dict, cfg: dict) -> int:
    p = cfg["llm"]["points"]
    material = str(score.get("material")).lower() == "true"
    d = int(score["direction"])
    if material and d == 2:
        return p["material_direction_2"]
    if d == 1:
        return p["direction_1"]
    return p["other"]


def v1_release(bar_releases: pd.DataFrame, scores: pd.DataFrame, cfg: dict) -> dict:
    """LLM points and block decision for one stock's releases on the signal bar.
    The most positive release decides the points, the most negative decides blocking.
    `scores` must be usable scores only (status ok, this prompt version, no backfill)."""
    if bar_releases.empty or scores.empty:
        return {"points": 0, "block": False, "scored_ids": []}
    s = scores[scores["release_id"].isin(set(bar_releases["release_id"]))]
    if s.empty:
        return {"points": 0, "block": False, "scored_ids": []}
    recs = s.to_dict("records")
    points = max(llm_points(r, cfg) for r in recs)
    lim = cfg["llm"]["block_if_direction_lte"]
    block = any(str(r["material"]).lower() == "true" and int(r["direction"]) <= lim for r in recs)
    return {"points": points, "block": block, "scored_ids": [r["release_id"] for r in recs]}


def with_release_points(pts: dict, release_pts: int, mcap: float | None, cfg: dict) -> dict:
    """v0's points with the release component replaced; small cap recomputed (it only
    amplifies an existing signal)."""
    out = {**pts, "release": release_pts, "small_cap": 0}
    if sum(out.values()) > 0 and mcap and mcap < cfg["thresholds"]["small_cap_max_sek"]:
        out["small_cap"] = 1
    return out


def strategy_scores(
    row: dict,
    releases: pd.DataFrame,
    scores: pd.DataFrame,
    cfg: dict,
    as_of: dt.datetime,
) -> dict:
    """score_v0 / score_v0b / score_v1, block_buy and the releases behind them, for one scanner row.
    `as_of`: releases published from then on don't count (as_of_for_bar)."""
    bar = releases_for_bar(releases, row["ticker"], row["date"], as_of)
    bar = bar[[not is_routine(t, cfg) for t in bar["title"]]] if not bar.empty else bar
    pts = row["points"]
    v0b = with_release_points(pts, RELEASE_POINTS_V0 if not bar.empty else 0, row.get("market_cap"), cfg)
    v1 = v1_release(bar, scores, cfg)
    v1_pts = with_release_points(pts, v1["points"], row.get("market_cap"), cfg)
    return {
        "score_v0": sum(pts.values()),
        "score_v0b": sum(v0b.values()),
        "score_v1": sum(v1_pts.values()),
        "block_buy": v1["block"],
        "release_ids": list(bar["release_id"]) if not bar.empty else [],
        "llm_release_ids": v1["scored_ids"],
    }
