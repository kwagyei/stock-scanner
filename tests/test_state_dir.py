"""SCANNER_STATE_DIR moves every generated file out of the code folder; SCANNER_CODE_URL makes
the generated pages link back to the code repository."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fund-watch"))


@pytest.fixture
def reloaded(monkeypatch, tmp_path):
    """Modules re-imported with the env vars set; restored afterwards."""
    names = ["scanner", "evaluate", "llm_score", "fund_watch"]

    def load(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return {n: importlib.reload(importlib.import_module(n)) for n in names}

    yield load
    monkeypatch.delenv("SCANNER_STATE_DIR", raising=False)
    monkeypatch.delenv("SCANNER_CODE_URL", raising=False)
    for n in names:
        importlib.reload(importlib.import_module(n))


def test_default_state_is_the_code_folder(reloaded, monkeypatch):
    monkeypatch.delenv("SCANNER_STATE_DIR", raising=False)
    m = reloaded()
    assert m["scanner"].DATA_DIR == ROOT / "data" and m["scanner"].DASHBOARD == ROOT / "DASHBOARD.md"
    assert m["fund_watch"].DATA_DIR == ROOT / "fund-watch" / "data"


def test_state_dir_moves_all_outputs(reloaded, tmp_path):
    m = reloaded(SCANNER_STATE_DIR=str(tmp_path))
    sc, ev, ls, fw = m["scanner"], m["evaluate"], m["llm_score"], m["fund_watch"]
    assert tmp_path / "data" == sc.DATA_DIR and tmp_path / "reports" == sc.REPORT_DIR
    assert tmp_path / "DASHBOARD.md" == sc.DASHBOARD
    assert tmp_path / "fund-watch" / "data" / "fund_watch_latest.json" == sc.FUND_SUMMARY
    assert tmp_path / "data" / "paper_trades.csv" == ev.PAPER_LOG
    assert tmp_path / "data" / "llm_scores.csv" == ls.SCORES_PATH
    assert tmp_path / "data" / "used_prompts.json" == ls.PROMPT_REGISTRY
    assert ls.PROMPT_DIR == ROOT / "prompts"  # prompts are code, not state
    assert tmp_path / "fund-watch" / "data" == fw.DATA_DIR
    assert fw.SUMMARY_FILE == sc.FUND_SUMMARY  # the dashboard finds the fund watch
    # code stays where it is
    assert sc.ROOT == ROOT


def test_code_links(reloaded):
    sc = reloaded()["scanner"]
    assert sc.code_link("README.md") == "README.md"
    assert sc.code_link("README.md", "reports/evaluation.md") == "../README.md"
    sc = reloaded(SCANNER_CODE_URL="https://github.com/u/stock-scanner/blob/main/")["scanner"]
    assert sc.code_link("config.yaml", "reports/x.md") == "https://github.com/u/stock-scanner/blob/main/config.yaml"
    assert "https://github.com/u/stock-scanner/blob/main/watchlist.csv" in sc._nav("reports/latest.md")


def test_fund_watch_creates_a_fresh_state_folder(reloaded, tmp_path, monkeypatch):
    """A brand-new state folder (no fund-watch/data yet) must not crash the first run."""
    fw = reloaded(SCANNER_STATE_DIR=str(tmp_path / "fresh"))["fund_watch"]
    monkeypatch.setattr(fw, "fetch_feed", lambda q, c: [])
    monkeypatch.setattr(fw.time, "sleep", lambda s: None)
    assert fw.main(["--no-email"]) == 0
    assert (tmp_path / "fresh" / "fund-watch" / "data" / "fund_watch_latest.json").exists()
    assert (tmp_path / "fresh" / "fund-watch" / "reports" / "fund_watch_latest.md").exists()
