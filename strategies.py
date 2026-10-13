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
from zoneinfo import ZoneInfo

import pandas as pd

STOCKHOLM = ZoneInfo("Europe/Stockholm")
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


def _local(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(STOCKHOLM)


def open_at(day) -> pd.Timestamp:
    """09:00 Stockholm on `day`, as an aware timestamp."""
    d = pd.Timestamp(day).date()
    return pd.Timestamp(dt.datetime.combine(d, OPEN), tz=STOCKHOLM)


def entry_day_for(published_at, trading_days) -> pd.Timestamp | None:
    """First trading day whose 09:00 open is after the release (None if not in the data yet)."""
    t = _local(published_at)
    for d in sorted(pd.DatetimeIndex(trading_days)):
        if open_at(d) > t:
            return pd.Timestamp(d).normalize()
    return None


def signal_bar_for(published_at, trading_days) -> pd.Timestamp | None:
    """The trading day before the entry day (None if the entry day is unknown or is the first day)."""
    days = sorted(pd.DatetimeIndex(trading_days).normalize())
    entry = entry_day_for(published_at, days)
    if entry is None:
        return None
    i = days.index(entry)
    return days[i - 1] if i > 0 else None


def releases_for_bar(
    releases: pd.DataFrame, ticker: str, bar_date, now: dt.datetime, bar_complete_today: bool = True
) -> pd.DataFrame:
    """Primary releases for `ticker` that belong to signal bar `bar_date` (the latest bar in the
    data). Releases from 09:00 on that day onwards count. A session that is running right now
    (a weekday after the bar, past 09:00, before today's bar is complete) is not part of the
    window: its releases belong to today's bar, scored at the evening run."""
    if releases.empty:
        return releases
    lo = open_at(bar_date)
    now_l = _local(now)
    hi = now_l
    if (
        now_l.date() > pd.Timestamp(bar_date).date()
        and now_l.weekday() < 5
        and now_l.time() >= OPEN
        and not bar_complete_today
    ):
        hi = open_at(now_l.date())
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


def with_release_points(pts: dict, release_pts: int, mcap, cfg: dict) -> dict:
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
    now: dt.datetime,
    bar_complete_today: bool = True,
) -> dict:
    """score_v0 / score_v0b / score_v1, block_buy and the releases behind them, for one scanner row."""
    bar = releases_for_bar(releases, row["ticker"], row["date"], now, bar_complete_today)
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
