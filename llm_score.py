"""
LLM scoring of company press releases (data/releases.csv -> data/llm_scores.csv).

Each primary release published on or after llm.score_from_date is scored once per
prompt_version by Claude, which returns JSON matching SCHEMA (validated here; one retry on
invalid output, then status=error). The release is sent in its original language; the
rationale comes back in English. The API key is read from ANTHROPIC_API_KEY by the SDK and
is never printed or stored.

    python llm_score.py                 # score new releases (the scanner also does this)
    python llm_score.py --backfill      # also older releases, marked is_backfill=true
                                        # (excluded from evaluation and paper trading)

Statuses: ok, error (invalid output twice, or a refusal; final), api_error (API unreachable,
rate limited or no key; retried next run), skipped_routine (title on release_ignore_titles;
no call made).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the SDK is imported lazily at run time, so a run with LLM scoring off never loads it
    import anthropic

import pandas as pd

import news_sources as ns

ROOT = Path(__file__).resolve().parent
PROMPT_DIR = ROOT / "prompts"
DATA_DIR = Path(os.environ.get("SCANNER_STATE_DIR") or ROOT) / "data"  # see scanner.STATE_DIR
SCORES_PATH = DATA_DIR / "llm_scores.csv"
PROMPT_REGISTRY = DATA_DIR / "used_prompts.json"  # sha256 of each prompt version once used live
UTC = dt.UTC

EVENT_TYPES = [
    "report",
    "order_contract",
    "guidance",
    "capital_raise",
    "insider",
    "m_and_a",
    "regulatory_clinical",
    "management",
    "other",
]
SCHEMA = {
    "type": "object",
    "properties": {
        "event_type": {"type": "string", "enum": EVENT_TYPES},
        "direction": {"type": "integer", "enum": [-2, -1, 0, 1, 2], "description": "expected short-term price impact"},
        "surprise": {
            "type": "integer",
            "enum": [0, 1, 2, 3],
            "description": "how unexpected vs likely market expectations",
        },
        "material": {"type": "boolean", "description": "would a reasonable investor consider this price-relevant"},
        "confidence": {"type": "number", "description": "0 to 1"},
        "rationale": {"type": "string", "description": "in English, at most 30 words"},
    },
    "required": ["event_type", "direction", "surprise", "material", "confidence", "rationale"],
    "additionalProperties": False,
}
# The user turn is part of a prompt version too (it is hashed with the system prompt).
USER_TEMPLATE = (
    "Company: {name} ({ticker})\nPublished (UTC): {published_at}\nTitle: {title}\n\n"
    "{body}\n\n"
    "Score this release in its original language. Write the rationale in English."
)
SCORE_COLUMNS = [
    "release_id",
    "ticker",
    "published_at",
    "scored_at",
    "model",
    "prompt_version",
    "event_type",
    "direction",
    "surprise",
    "material",
    "confidence",
    "rationale",
    "status",
    "error",
    "is_backfill",
    "input_tokens",
    "output_tokens",
]
FINAL_STATUSES = {"ok", "error", "skipped_routine"}
MAX_TOKENS = 8000


class PromptChanged(RuntimeError):
    pass


# --------------------------------------------------------------------------- prompt


def load_prompt(version: str) -> str:
    return (PROMPT_DIR / f"score_{version}.md").read_text(encoding="utf-8")


def prompt_fingerprint(prompt: str, effort: str) -> str:
    return hashlib.sha256(f"{prompt}\n--\n{USER_TEMPLATE}\n--\neffort={effort}".encode()).hexdigest()


def check_prompt_unchanged(version: str, prompt: str, effort: str, register: bool) -> None:
    """Refuse to score with a prompt whose text (or user template / effort) changed after it was
    used live. The first live use records its fingerprint in data/used_prompts.json."""
    reg = json.loads(PROMPT_REGISTRY.read_text()) if PROMPT_REGISTRY.exists() else {}
    fp = prompt_fingerprint(prompt, effort)
    if version in reg and reg[version] != fp:
        raise PromptChanged(
            f"prompt {version} changed after it was used live; add a new version "
            f"(prompts/score_v<N>.md) instead of editing it"
        )
    if register and version not in reg:
        reg[version] = fp
        PROMPT_REGISTRY.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- validation


def validate(obj: object) -> str | None:
    """None if obj matches SCHEMA (plus the ranges the API schema can't express), else the reason."""
    if not isinstance(obj, dict):
        return "not an object"
    if set(obj) != set(SCHEMA["required"]):
        return f"keys {sorted(obj)}"
    if obj["event_type"] not in EVENT_TYPES:
        return f"event_type {obj['event_type']!r}"
    for k, lo, hi in (("direction", -2, 2), ("surprise", 0, 3)):
        v = obj[k]
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            return f"{k} {v!r}"
    if not isinstance(obj["material"], bool):
        return f"material {obj['material']!r}"
    c = obj["confidence"]
    if isinstance(c, bool) or not isinstance(c, (int, float)) or not 0 <= c <= 1:
        return f"confidence {c!r}"
    r = obj["rationale"]
    if not isinstance(r, str) or not r.strip() or len(r.split()) > 30:
        return "rationale missing or over 30 words"
    return None


# --------------------------------------------------------------------------- API call


def make_client() -> anthropic.Anthropic | None:
    """Anthropic client; the SDK reads ANTHROPIC_API_KEY itself. None if the key is not set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    return anthropic.Anthropic()


def _ask(client: anthropic.Anthropic, cfg: dict, prompt: str, release: dict, name: str) -> anthropic.types.Message:
    llm = cfg["llm"]
    return client.messages.create(
        model=llm["model"],
        max_tokens=MAX_TOKENS,
        system=prompt,
        output_config={"effort": llm.get("effort", "medium"), "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    name=name or release["ticker"],
                    ticker=release["ticker"],
                    published_at=release["published_at"],
                    title=release["title"],
                    body=(release.get("body_text") or "")[: llm["max_body_chars"]] or "(no body text)",
                ),
            }
        ],
    )


def score_release(client: anthropic.Anthropic, cfg: dict, prompt: str, release: dict, name: str = "") -> dict:
    """Score one release: up to two attempts on invalid output. Never raises for API errors."""
    import anthropic

    usage_in = usage_out = 0
    reason = ""
    for _ in range(2):
        try:
            resp = _ask(client, cfg, prompt, release, name)
        except anthropic.APIStatusError as exc:
            if 400 <= exc.status_code < 500 and exc.status_code not in (408, 409, 429):
                return {
                    "status": "error",
                    "error": f"API {exc.status_code}: {exc.message}"[:200],
                    "input_tokens": usage_in,
                    "output_tokens": usage_out,
                }
            return {
                "status": "api_error",
                "error": f"API {exc.status_code}"[:200],
                "input_tokens": usage_in,
                "output_tokens": usage_out,
            }
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
            return {
                "status": "api_error",
                "error": type(exc).__name__,
                "input_tokens": usage_in,
                "output_tokens": usage_out,
            }
        usage_in += getattr(resp.usage, "input_tokens", 0) or 0
        usage_out += getattr(resp.usage, "output_tokens", 0) or 0
        if resp.stop_reason == "refusal":
            return {"status": "error", "error": "refusal", "input_tokens": usage_in, "output_tokens": usage_out}
        if resp.stop_reason == "max_tokens":
            reason = "hit max_tokens"
            continue
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            reason = "invalid JSON"
            continue
        reason = validate(obj)
        if reason is None:
            return {**obj, "status": "ok", "error": "", "input_tokens": usage_in, "output_tokens": usage_out}
    return {
        "status": "error",
        "error": f"invalid output twice ({reason})"[:200],
        "input_tokens": usage_in,
        "output_tokens": usage_out,
    }


# --------------------------------------------------------------------------- store


def load_scores(path: Path = None) -> pd.DataFrame:
    path = Path(path or SCORES_PATH)
    if not path.exists():
        return pd.DataFrame(columns=SCORE_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in SCORE_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[SCORE_COLUMNS]


def save_scores(df: pd.DataFrame, path: Path = None) -> None:
    df[SCORE_COLUMNS].to_csv(Path(path or SCORES_PATH), index=False)


def usable_scores(scores: pd.DataFrame, prompt_version: str) -> pd.DataFrame:
    """Scores the strategies and evaluation may use: status ok, this prompt version, no backfill."""
    return scores[
        (scores["status"] == "ok")
        & (scores["prompt_version"] == prompt_version)
        & (scores["is_backfill"].astype(str).str.lower() != "true")
    ]


def is_routine(title: str, cfg: dict) -> bool:
    t = (title or "").lower()
    return any(p.lower() in t for p in cfg.get("release_ignore_titles", []) or [])


def to_score(releases: pd.DataFrame, scores: pd.DataFrame, cfg: dict, backfill: bool) -> pd.DataFrame:
    """Primary releases still needing a score for this prompt version, oldest first."""
    version = cfg["llm"]["prompt_version"]
    done = set(scores.loc[(scores["prompt_version"] == version) & scores["status"].isin(FINAL_STATUSES), "release_id"])
    todo = releases[(releases["source_type"] == "primary") & ~releases["release_id"].isin(done)]
    if not backfill:
        todo = todo[todo["published_at"] >= str(cfg["llm"]["score_from_date"])]
    return todo.sort_values("published_at", kind="stable")


def run(
    cfg: dict,
    releases: pd.DataFrame,
    scores: pd.DataFrame,
    backfill: bool = False,
    client: anthropic.Anthropic | None = None,
    names: dict | None = None,
    now: dt.datetime | None = None,
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """Score what's pending. Returns (all score rows, rows written this run, warnings)."""
    llm = cfg["llm"]
    now = now or dt.datetime.now(UTC)
    names = names or {}
    warnings: list[str] = []
    todo = to_score(releases, scores, cfg, backfill)
    if todo.empty:
        return scores, [], warnings
    prompt = load_prompt(llm["prompt_version"])
    check_prompt_unchanged(llm["prompt_version"], prompt, llm.get("effort", "medium"), register=not backfill)
    score_from = str(llm["score_from_date"])
    new, calls = [], 0
    for _, rel in todo.iterrows():
        rel = rel.to_dict()
        base = {
            "release_id": rel["release_id"],
            "ticker": rel["ticker"],
            "published_at": rel["published_at"],
            "scored_at": ns._utc_iso(now),
            "model": llm["model"],
            "prompt_version": llm["prompt_version"],
            "is_backfill": str(rel["published_at"] < score_from).lower(),
        }
        if llm.get("skip_routine", True) and is_routine(rel["title"], cfg):
            new.append({**base, "status": "skipped_routine", "error": "", "input_tokens": 0, "output_tokens": 0})
            continue
        if client is None:
            client = make_client()
            if client is None:
                warnings.append(f"ANTHROPIC_API_KEY not set: {len(todo) - len(new)} release(s) not scored")
                break
        if calls >= llm["max_calls_per_run"]:
            warnings.append(
                f"max_calls_per_run ({llm['max_calls_per_run']}) reached: "
                f"{len(todo) - len(new)} release(s) left for the next run"
            )
            break
        calls += 1
        result = score_release(client, cfg, prompt, rel, names.get(rel["ticker"], ""))
        new.append({**base, **result})
        if result["status"] != "ok":
            warnings.append(f"{rel['ticker']} '{rel['title'][:60]}': {result['status']} ({result['error']})")
    if new:
        retried = {r["release_id"] for r in new}
        keep = scores[~((scores["prompt_version"] == llm["prompt_version"]) & scores["release_id"].isin(retried))]
        scores = pd.concat([keep, pd.DataFrame(new).reindex(columns=SCORE_COLUMNS)], ignore_index=True)
    return scores, new, warnings


def main(argv: list[str] | None = None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument(
        "--backfill", action="store_true", help="also score releases before llm.score_from_date (is_backfill=true)"
    )
    args = ap.parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not (cfg.get("llm") or {}).get("enabled", True):
        print("[info] llm.enabled is false in config.yaml: nothing scored, no API calls")
        return 0
    releases = ns.load_releases(DATA_DIR / "releases.csv")
    scores, new, warnings = run(cfg, releases, load_scores(), backfill=args.backfill)
    if new:
        save_scores(scores)
    for w in warnings:
        print(f"[warn] {w}", file=sys.stderr)
    ok = sum(r["status"] == "ok" for r in new)
    tokens = sum(int(r.get("input_tokens") or 0) + int(r.get("output_tokens") or 0) for r in new)
    print(f"[info] llm_score: {len(new)} row(s) written, {ok} ok, {tokens} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
