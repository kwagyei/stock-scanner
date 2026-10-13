"""The deployed schedule must agree with the scanner's own clock rules in summer and in winter.

GitHub cron runs in UTC, Stockholm doesn't: an evening run at 16:52 UTC is 18:52 in summer
but 17:52 in winter, before today's bar counts as complete (complete_bar_after). Such a run
scores yesterday's bar and then blocks the evening slot for the run that would have scored
today's. These tests pin the cron times, the slot windows and the config to each other.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCAN_WORKFLOW = ROOT / "deploy" / "data-repo" / ".github" / "workflows" / "scan.yml"
STOCKHOLM = ZoneInfo("Europe/Stockholm")
SUMMER, WINTER = dt.date(2026, 7, 15), dt.date(2026, 11, 18)  # CEST (UTC+2), CET (UTC+1)


def _cfg() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def _workflow() -> dict:
    return yaml.safe_load(SCAN_WORKFLOW.read_text())


def _hm(text: str) -> dt.time:
    h, m = (int(x) for x in text.split(":"))
    return dt.time(h, m)


def _local(cron: str, day: dt.date) -> dt.time:
    minute, hour = (int(x) for x in cron.split()[:2])
    utc = dt.datetime.combine(day, dt.time(hour, minute), dt.UTC)
    return utc.astimezone(STOCKHOLM).time()


def _crons() -> list[str]:
    return [c["cron"] for c in _workflow()[True]["schedule"]]  # PyYAML reads the key `on` as True


def _slot_script() -> str:
    return next(s["run"] for s in _workflow()["jobs"]["slot"]["steps"] if s.get("id") == "check")


def _evening_crons() -> list[str]:
    """The crons the slot step maps to the evening slot."""
    env = next(s for s in _workflow()["jobs"]["slot"]["steps"] if s.get("id") == "check")["env"]["SLOT"]
    return re.findall(r"github\.event\.schedule == '([^']+)' && 'evening'", env)


def test_every_scheduled_run_maps_to_a_known_cron():
    assert _evening_crons() and set(_evening_crons()) <= set(_crons())


@pytest.mark.parametrize("day", [SUMMER, WINTER], ids=["summer", "winter"])
def test_evening_runs_land_after_todays_bar_is_complete(day):
    cutoff = _hm(_cfg()["complete_bar_after"])
    for cron in _evening_crons():
        assert _local(cron, day) >= cutoff, f"{cron} is {_local(cron, day)} Stockholm, before {cutoff}"


@pytest.mark.parametrize("day", [SUMMER, WINTER], ids=["summer", "winter"])
def test_morning_runs_land_before_the_open(day):
    for cron in set(_crons()) - set(_evening_crons()):
        assert _local(cron, day) < dt.time(9, 0), f"{cron} is {_local(cron, day)} Stockholm"


def test_evening_slot_starts_at_the_bar_cutoff():
    """A scan committed before the cutoff (on yesterday's bar) must not count as the evening
    scan, and an evening-slot run before the cutoff must skip instead of scanning."""
    script = _slot_script()
    cutoff = _cfg()["complete_bar_after"]
    assert f'evening) from="{cutoff}"' in script
    assert "too early for the evening slot" in script


def test_weekly_summary_waits_for_the_complete_friday_bar():
    cfg = _cfg()
    assert dt.time(cfg["weekly_summary"]["after_hour"]) >= _hm(cfg["complete_bar_after"])
