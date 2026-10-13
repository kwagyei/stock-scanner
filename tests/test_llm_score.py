"""Offline tests for llm_score.py. The Anthropic client is faked; sockets are blocked."""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pandas as pd
import pytest

import llm_score as ls
import news_sources as ns
import scanner

UTC = dt.UTC
NOW = dt.datetime(2026, 10, 9, 6, 0, tzinfo=UTC)
GOOD = {
    "event_type": "order_contract",
    "direction": 2,
    "surprise": 2,
    "material": True,
    "confidence": 0.8,
    "rationale": "Order worth a third of annual revenue.",
}


@pytest.fixture(autouse=True)
def tmp_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "PROMPT_REGISTRY", tmp_path / "used_prompts.json")
    monkeypatch.setattr(ls, "SCORES_PATH", tmp_path / "llm_scores.csv")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return tmp_path


@pytest.fixture
def cfg():
    return scanner.load_config(scanner.ROOT / "config.yaml")


def resp(payload, stop="end_turn", tin=100, tout=20):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        stop_reason=stop,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=tin, output_tokens=tout),
    )


class FakeClient:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a


def release(rid="r1", ticker="ELTEL.ST", ts="2026-10-08T06:30:00Z", title="Eltel vinner order", source_type="primary"):
    return {
        "release_id": rid,
        "ticker": ticker,
        "source": "mfn",
        "source_type": source_type,
        "published_at": ts,
        "title": title,
        "url": f"https://mfn.se/a/{rid}",
        "language": "sv",
        "body_text": "Eltel har fått en order.",
        "fetched_at": ts,
    }


def releases(*rows):
    return pd.DataFrame(list(rows), columns=ns.RELEASE_COLUMNS)


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "change",
    [
        {"direction": 3},
        {"direction": 1.5},
        {"direction": True},
        {"surprise": -1},
        {"event_type": "rumour"},
        {"material": "yes"},
        {"confidence": 1.2},
        {"rationale": " ".join(["word"] * 31)},
        {"rationale": ""},
        {"extra": 1},
    ],
)
def test_validate_rejects_out_of_schema(change):
    assert ls.validate({**GOOD, **change}) is not None


def test_validate_accepts_good_and_drop_key_fails():
    assert ls.validate(GOOD) is None
    assert ls.validate({k: v for k, v in GOOD.items() if k != "surprise"}) is not None


def test_schema_matches_required_fields():
    assert set(ls.SCHEMA["properties"]) == set(ls.SCHEMA["required"])
    assert ls.SCHEMA["additionalProperties"] is False


# --------------------------------------------------------------------------- scoring + retry


def test_score_release_ok_and_request_shape(cfg):
    client = FakeClient(resp(GOOD))
    out = ls.score_release(client, cfg, "SYSTEM", release(), "Eltel")
    assert out["status"] == "ok" and out["direction"] == 2 and out["input_tokens"] == 100
    kw = client.calls[0]
    assert kw["model"] == cfg["llm"]["model"] and kw["system"] == "SYSTEM"
    assert kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["output_config"]["effort"] == cfg["llm"]["effort"]
    assert "Eltel har fått en order." in kw["messages"][0]["content"]
    assert "rationale in English" in kw["messages"][0]["content"]


def test_score_release_retries_once_then_error(cfg):
    client = FakeClient(resp("not json"), resp(GOOD))
    assert ls.score_release(client, cfg, "S", release())["status"] == "ok"
    client = FakeClient(resp({**GOOD, "direction": 5}), resp("{"))
    out = ls.score_release(client, cfg, "S", release())
    assert out["status"] == "error" and "invalid output twice" in out["error"]
    assert out["input_tokens"] == 200 and len(client.calls) == 2


def test_refusal_is_error_without_retry(cfg):
    client = FakeClient(resp("", stop="refusal"))
    out = ls.score_release(client, cfg, "S", release())
    assert out["status"] == "error" and out["error"] == "refusal" and len(client.calls) == 1


def test_max_tokens_counts_as_invalid(cfg):
    client = FakeClient(resp('{"event', stop="max_tokens"), resp(GOOD))
    assert ls.score_release(client, cfg, "S", release())["status"] == "ok"


def test_api_errors_are_recorded_not_raised(cfg):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    conn = anthropic.APIConnectionError(request=req)
    out = ls.score_release(FakeClient(conn), cfg, "S", release())
    assert out["status"] == "api_error"
    bad = anthropic.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    out = ls.score_release(FakeClient(bad), cfg, "S", release())
    assert out["status"] == "error" and out["error"].startswith("API 400")


# --------------------------------------------------------------------------- run: idempotency, dates, cap


def test_run_is_idempotent_per_release_and_prompt_version(cfg):
    rel = releases(release())
    client = FakeClient(resp(GOOD))
    scores, new, warn = ls.run(cfg, rel, ls.load_scores(), client=client, now=NOW)
    assert len(new) == 1 and warn == []
    scores2, new2, _ = ls.run(cfg, rel, scores, client=FakeClient(), now=NOW)
    assert new2 == [] and len(scores2) == 1
    # a new prompt version scores it again
    cfg2 = {**cfg, "llm": {**cfg["llm"], "prompt_version": "v1"}}
    assert ls.to_score(rel, scores2.assign(prompt_version="v0x"), cfg2, False).shape[0] == 1


def test_api_error_rows_are_retried_and_replaced(cfg):
    rel = releases(release())
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    scores, new, warn = ls.run(
        cfg, rel, ls.load_scores(), client=FakeClient(anthropic.APIConnectionError(request=req)), now=NOW
    )
    assert new[0]["status"] == "api_error" and warn
    scores, new, _ = ls.run(cfg, rel, scores, client=FakeClient(resp(GOOD)), now=NOW)
    assert list(scores["status"]) == ["ok"]


def test_score_from_date_and_backfill(cfg):
    old = release("old", ts="2026-09-30T06:00:00Z")
    live = release("live", ts=f"{cfg['llm']['score_from_date']}T07:00:00Z")
    agg = release("agg", source_type="aggregator")
    rel = releases(old, live, agg)
    scores, new, _ = ls.run(cfg, rel, ls.load_scores(), client=FakeClient(resp(GOOD)), now=NOW)
    assert [r["release_id"] for r in new] == ["live"] and new[0]["is_backfill"] == "false"
    scores, new, _ = ls.run(cfg, rel, scores, backfill=True, client=FakeClient(resp(GOOD)), now=NOW)
    assert [r["release_id"] for r in new] == ["old"] and new[0]["is_backfill"] == "true"
    usable = ls.usable_scores(scores, cfg["llm"]["prompt_version"])
    assert list(usable["release_id"]) == ["live"]


def test_routine_titles_skip_the_call(cfg):
    routine = release("r", title="Antalet aktier och röster i Eltel AB per den 30 september 2026")
    scores, new, _ = ls.run(cfg, releases(routine), ls.load_scores(), client=FakeClient(), now=NOW)
    assert new[0]["status"] == "skipped_routine"


def test_cost_cap_and_missing_key(cfg):
    cfg = {**cfg, "llm": {**cfg["llm"], "max_calls_per_run": 1}}
    rel = releases(release("a"), release("b", ts="2026-10-08T08:00:00Z"))
    scores, new, warn = ls.run(cfg, rel, ls.load_scores(), client=FakeClient(resp(GOOD)), now=NOW)
    assert len(new) == 1 and "max_calls_per_run" in warn[0]
    scores, new, warn = ls.run(cfg, rel, scores, now=NOW)  # no client, no key
    assert new == [] and "ANTHROPIC_API_KEY not set" in warn[0]


def test_save_load_roundtrip(cfg, tmp_registry):
    scores, new, _ = ls.run(cfg, releases(release()), ls.load_scores(), client=FakeClient(resp(GOOD)), now=NOW)
    ls.save_scores(scores)
    back = ls.load_scores()
    assert list(back.columns) == ls.SCORE_COLUMNS and back.iloc[0]["direction"] == "2"
    assert back.iloc[0]["scored_at"] == "2026-10-09T06:00:00Z"


# --------------------------------------------------------------------------- prompt guard


def test_prompt_guard_blocks_edits_after_live_use(cfg, monkeypatch):
    ls.run(cfg, releases(release()), ls.load_scores(), client=FakeClient(resp(GOOD)), now=NOW)
    assert "v1" in json.loads(ls.PROMPT_REGISTRY.read_text())
    monkeypatch.setattr(ls, "load_prompt", lambda v: "edited prompt")
    with pytest.raises(ls.PromptChanged):
        ls.run(cfg, releases(release("r2")), ls.load_scores(), client=FakeClient(resp(GOOD)), now=NOW)


def test_prompt_file_exists_for_configured_version(cfg):
    assert ls.load_prompt(cfg["llm"]["prompt_version"]).startswith("You are an equity analyst")
