"""Offline tests for fund_watch.py. Feeds are mocked; sockets are blocked."""
import datetime as dt
import json
import socket

import pytest

import fund_watch as fw

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 7, 5, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def guard(*a, **k):
        raise RuntimeError("network access in tests")
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(fw.time, "sleep", lambda s: None)
    monkeypatch.setattr(fw, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(fw, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(fw, "SEEN_FILE", tmp_path / "data" / "fund_watch_seen.json")
    monkeypatch.setattr(fw, "LOG_FILE", tmp_path / "data" / "fund_watch_log.csv")
    monkeypatch.setattr(fw, "SUMMARY_FILE", tmp_path / "data" / "fund_watch_latest.json")


@pytest.fixture
def cfg():
    return fw.load_config(fw.ROOT / "fund_watch.yaml")


def item(title, summary="", hours_ago=1, source="Reuters"):
    return {"title": f"{title} - {source}", "link": "http://x", "source": source,
            "ts": NOW - dt.timedelta(hours=hours_ago), "summary": summary}


def classify(title, cfg, summary=""):
    return fw.classify(item(title, summary), fw.compile_categories(cfg), fw.build_name_sets(cfg), cfg)


# --------------------------------------------------------------------------- whole-word matching

def _war_cfg(cfg):
    c = dict(cfg)
    c["risk_categories"] = {"geopolitics": {"severity": 3, "keywords": ["war", "PLA", "ban"]}}
    return c


@pytest.mark.parametrize("title,hit", [
    ("Global warming threatens chip fabs", False),      # "warming" must not match "war"
    ("Warships sail near Taiwan", False),
    ("Software wares on sale", False),
    ("Fear of war grows", True),
    ("War! Markets react", True),
    ("Post-war boom", False),                            # hyphenated compound is not the word
    ("Explained: the plan for Taiwan", False),          # "PLA" inside "plan" / "explained"
    ("PLA aircraft cross median line", True),
    ("Bangkok banks rally", False), ("Urban planning", False),
    ("US weighs ban on chips", True),
])
def test_whole_word_matching(cfg, title, hit):
    assert (classify(title, _war_cfg(cfg)) is not None) is hit


def test_matching_is_case_insensitive_and_phrases_need_all_words(cfg):
    assert classify("TSMC HALTS PRODUCTION after quake", cfg)["categories"]["infrastructure"] == ["halts production"]
    assert classify("Production lines halt briefly", cfg) is None


# --------------------------------------------------------------------------- classification

def test_unrelated_news_is_dropped(cfg):
    assert classify("TSMC opens new visitor centre", cfg) is None


@pytest.mark.parametrize("title,category,severity", [
    ("China announces blockade drills around Taiwan", "geopolitics", 3),
    ("US adds firms to entity list over chip exports", "trade_restrictions", 3),
    ("Strong earthquake jolts Hsinchu science park", "natural_disaster", 3),
    ("Fire broke out at a Kaohsiung plant", "infrastructure", 2),
    ("Workers stage walkout at supplier", "labour", 2),
    ("Apple cuts orders for new phones", "demand_and_customers", 2),
    ("MediaTek cuts guidance on weak handset demand", "company_results", 2),
    ("Prosecutors indicted two engineers over trade secrets", "legal_regulatory", 2),
    ("Ransomware hits chip packaging firm", "cyber_operations", 2),
    ("Taiwan stocks plunge on foreign selling", "market_stress", 1),
    ("JPMorgan Taiwan Fund suspends dealing", "fund_specific", 3),
])
def test_categories(cfg, title, category, severity):
    c = classify(title, cfg)
    assert category in c["categories"] and c["severity"] >= severity


def test_severity_is_the_highest_matching_category(cfg):
    c = classify("Taiwan stocks plunge as China announces blockade", cfg)
    assert set(c["categories"]) == {"market_stress", "geopolitics"} and c["severity"] == 3


@pytest.mark.parametrize("title", ["Magnitude 4.5 earthquake shakes southern Taiwan",
                                   "Light mag. 3.1 earthquake - Fujian", "An Earthquake in the Mac World",
                                   "War game highlights Taiwan response", "Microsoft fires CEO"])
def test_tuned_false_positives_stay_quiet(cfg, title):
    assert classify(title, cfg) is None


@pytest.mark.parametrize("title", ["Magnitude 6.4 earthquake strikes off Hualien",
                                   "Powerful quake rocks eastern Taiwan", "Mag. 7.1 quake near Taiwan"])
def test_strong_quakes_still_flag(cfg, title):
    assert "natural_disaster" in classify(title, cfg)["categories"]


# --------------------------------------------------------------------------- scoring

def test_scoring_bonuses(cfg):
    s = cfg["scoring"]
    base = classify("Strike vote at chip plant", cfg)                                # labour, 2
    assert (base["score"], base["entities"], base["relevant"]) == (2, [], False)
    tw = classify("Strike vote at Taiwan chip plant", cfg)
    assert tw["score"] == 2 + s["taiwan_bonus"] and tw["relevant"]
    top = classify("Strike vote at TSMC plant", cfg)                                  # top-5 holding
    assert top["score"] == 2 + s["holding_bonus"] and top["entities"] == ["TSMC"]
    alias = classify("Strike vote at Taiwan Semiconductor plant", cfg)
    assert alias["entities"] == ["TSMC"]                                              # alias -> holding name
    other = classify("Strike vote at Quanta plant", cfg)                              # watched, < 3% weight
    assert other["score"] == 2 + s["other_holding_bonus"] and other["entities"] == ["Quanta Computer"]
    both = classify("Strike at TSMC and Quanta plants", cfg)
    assert both["score"] == 2 + s["holding_bonus"]                                    # top bonus only, not both
    dep = classify("Strike vote at Nvidia supplier", cfg)
    assert dep["score"] == 2 + s["dependency_bonus"] and dep["entities"] == ["Nvidia"]
    full = classify("TSMC halts production in Taiwan, Nvidia hit", cfg)
    assert full["score"] == 2 + s["holding_bonus"] + s["dependency_bonus"] + s["taiwan_bonus"]


def test_relevance_gate_moves_unrelated_high_severity_to_watch(cfg):
    far = classify("Strong earthquake hits Peru", cfg)                                # HIGH, score 3, no link
    near = classify("Strong earthquake hits Taiwan", cfg)
    assert far["score"] == 3 and not far["relevant"]
    flagged, watch = fw.split_flagged([far, near], cfg)
    assert flagged == [near] and watch == [far]
    off = {**cfg, "scoring": {**cfg["scoring"], "require_relevance": False}}
    assert fw.split_flagged([far, near], off)[0] == [far, near]


def test_fund_name_counts_as_relevant(cfg):
    assert classify("JPM Taiwan Fund manager change announced", cfg)["relevant"]


# --------------------------------------------------------------------------- dedup

def test_item_key_ignores_source_suffix_case_and_punctuation():
    a = fw.item_key("TSMC halts production after quake - Reuters")
    assert a == fw.item_key("TSMC Halts Production After Quake! - Bloomberg.com")
    assert a != fw.item_key("TSMC resumes production after quake - Reuters")


# --------------------------------------------------------------------------- end to end: seen-store

def run(monkeypatch, feeds, *args):
    """feeds: query substring -> list of items; unmatched queries return nothing."""
    calls = []
    def fake(query, cfg):
        calls.append(query)
        return [dict(i) for key, items in feeds.items() if key in query for i in items]
    monkeypatch.setattr(fw, "fetch_feed", fake)
    monkeypatch.setattr(fw.dt, "datetime", _FrozenDatetime)
    assert fw.main(["--no-email", *args]) == 0
    return calls


class _FrozenDatetime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz else NOW.replace(tzinfo=None)


def _log():
    return fw.LOG_FILE.read_text().splitlines()[1:] if fw.LOG_FILE.exists() else []


def test_dedup_across_queries_and_seen_store(monkeypatch, cfg):
    quake = item("Strong earthquake halts TSMC fabs in Taiwan")
    dup = item("Strong earthquake halts TSMC fabs in Taiwan", source="Bloomberg")   # same story, other outlet
    calm = item("TSMC opens visitor centre")
    feeds = {"TSMC": [quake, calm], "Taiwan earthquake": [dup]}
    calls = run(monkeypatch, feeds)
    assert len(calls) == len(fw.build_queries(cfg))                                  # every search ran
    assert len(_log()) == 1                                                           # dup + unrelated removed
    seen = json.loads(fw.SEEN_FILE.read_text())
    assert len(seen) == 2                                                             # unrelated item marked seen too

    run(monkeypatch, feeds)                                                           # same news again
    assert len(_log()) == 1
    html = (fw.REPORT_DIR / "fund_watch_latest.html").read_text()
    assert "Nothing new above the flag threshold." in html


def test_all_flag_ignores_and_keeps_seen_store_and_log(monkeypatch):
    feeds = {"TSMC": [item("Strong earthquake halts TSMC fabs in Taiwan")]}
    run(monkeypatch, feeds)
    before = fw.SEEN_FILE.read_text()
    run(monkeypatch, feeds, "--all")
    assert "Strong earthquake halts TSMC" in (fw.REPORT_DIR / "fund_watch_latest.html").read_text()
    assert fw.SEEN_FILE.read_text() == before and len(_log()) == 1


def test_seen_store_retention(cfg):
    fw.DATA_DIR.mkdir()
    old = (NOW - dt.timedelta(days=31)).isoformat()
    fresh = (NOW - dt.timedelta(days=2)).isoformat()
    fw.save_seen({"a": old, "b": fresh}, cfg["news"]["seen_retention_days"], NOW)
    assert fw.load_seen() == {"b": fresh}


def test_corrupt_seen_store_starts_fresh():
    fw.DATA_DIR.mkdir()
    fw.SEEN_FILE.write_text("{not json")
    assert fw.load_seen() == {}


def test_ranking_and_flag_split(monkeypatch):
    feeds = {"TSMC": [item("TSMC halts production in Taiwan, Nvidia hit"),      # 2+2+1+1 = 6
                      item("Strike vote at Quanta plant"),                    # 2+1 = 3
                      item("Strong earthquake hits Peru")]}                   # 3, not relevant -> watch
    run(monkeypatch, feeds)
    rows = [r.split(",") for r in _log()]
    assert [r[2] for r in rows] == ["6", "3", "3"]
    html = (fw.REPORT_DIR / "fund_watch_latest.html").read_text()
    flagged, watch = html.split("<h2>Watch")
    assert "Peru" in watch and "Peru" not in flagged and "Quanta" in flagged


def test_feed_failure_warns(monkeypatch, capsys, cfg):
    failed = fw.feedparser.FeedParserDict(bozo=1, bozo_exception="SSL: CERTIFICATE_VERIFY_FAILED", entries=[])
    monkeypatch.setattr(fw.feedparser, "parse", lambda url, agent=None: failed)
    assert fw.fetch_feed('"TSMC"', cfg) == []
    assert "feed failed" in capsys.readouterr().err


def test_email_only_when_flagged(monkeypatch, cfg):
    sent = []
    monkeypatch.setattr(fw, "send_email", lambda html, subject: sent.append(subject))
    on = {**cfg, "email": {"enabled": True, "send_when": "flagged"}}
    monkeypatch.setattr(fw, "load_config", lambda path: on)
    monkeypatch.setattr(fw, "fetch_feed", lambda q, c: [])
    monkeypatch.setattr(fw.dt, "datetime", _FrozenDatetime)
    fw.main([])
    assert sent == []                                                            # nothing flagged: no mail
    monkeypatch.setattr(fw, "fetch_feed", lambda q, c: [item("China blockade of Taiwan begins")] if "Strait" in q else [])
    fw.main([])
    assert len(sent) == 1 and "1 flagged (1 high)" in sent[0]
    fw.main(["--no-email"])
    assert len(sent) == 1


# --------------------------------------------------------------------------- e-mail

SMTP_ENV = {"SMTP_HOST": "smtp.gmail.com", "SMTP_PORT": "587", "SMTP_USER": "alerts@example.test",
            "SMTP_PASS": "app-password", "EMAIL_TO": "me@example.test"}


class FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.calls = host, port, []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.calls.append("starttls")

    def login(self, user, pwd):
        self.calls.append(("login", user))

    def send_message(self, msg):
        self.calls.append(("send", msg["From"], msg["To"], msg["Subject"]))


def test_send_email_uses_starttls_and_login(monkeypatch):
    FakeSMTP.instances = []
    for k, v in SMTP_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(fw.smtplib, "SMTP", FakeSMTP)
    fw.send_email("<p>x</p>", "Subject")
    s = FakeSMTP.instances[0]
    assert (s.host, s.port) == ("smtp.gmail.com", 587)
    assert s.calls == ["starttls", ("login", "alerts@example.test"),
                       ("send", "alerts@example.test", "me@example.test", "Subject")]


def test_missing_settings_skip_normally_but_fail_a_test_email(monkeypatch, capsys):
    for k in SMTP_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SMTP_HOST", "smtp.gmail.com")
    fw.send_email("x", "s")
    assert "e-mail skipped, not set: SMTP_USER, SMTP_PASS, EMAIL_TO" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="SMTP_USER, SMTP_PASS, EMAIL_TO"):
        fw.send_email("x", "s", required=True)


def test_test_email_sends_even_with_nothing_flagged(monkeypatch):
    sent = []
    monkeypatch.setattr(fw, "send_email", lambda html, subject, required=False: sent.append((subject, required)))
    monkeypatch.setattr(fw, "fetch_feed", lambda q, c: [])
    monkeypatch.setattr(fw.dt, "datetime", _FrozenDatetime)
    fw.main(["--test-email"])
    assert sent == [("[TEST] Taiwan fund risk watch 07 Oct: 0 flagged (0 high)", True)]


def test_shipped_config_emails_only_when_flagged(cfg):
    assert cfg["email"] == {"enabled": True, "send_when": "flagged"}


# --------------------------------------------------------------------------- dashboard page + summary

def test_markdown_page(monkeypatch):
    feeds = {"TSMC": [item("TSMC | halts production in Taiwan, Nvidia hit"),       # flagged, HIGH? (infra = 2)
                      item("China blockade of Taiwan begins"),                     # flagged, HIGH
                      item("Strong earthquake hits Peru")]}                        # watch
    run(monkeypatch, feeds)
    md = (fw.REPORT_DIR / "fund_watch_latest.md").read_text()
    assert md.startswith("# 🛡️ Fund risk watch") and "(../../DASHBOARD.md)" in md
    assert "| **2** | **1** | **1** |" in md                                         # flagged, high, watch
    assert "> [!WARNING]\n> **1 high-severity item(s)**" in md
    assert "TSMC \\| halts production" in md and "🔴 HIGH" in md and "🟠 MEDIUM" in md
    flagged, watch = md.split("<details>")
    assert "Peru" in watch and "Peru" not in flagged
    assert md.count("<details>") == md.count("</details>")


def test_summary_written_on_normal_runs_only(monkeypatch):
    feeds = {"TSMC": [item("China blockade of Taiwan begins")]}
    run(monkeypatch, feeds, "--all")
    assert not fw.SUMMARY_FILE.exists()
    run(monkeypatch, feeds)
    s = json.loads(fw.SUMMARY_FILE.read_text())
    assert (s["n_flagged"], s["n_high"], s["n_watch"], s["run_utc"]) == (1, 1, 0, "2026-10-07T05:00+00:00")
    assert s["top"][0]["severity"] == "HIGH" and s["top"][0]["title"].startswith("China blockade")
    run(monkeypatch, feeds)                                                        # nothing new next time
    assert json.loads(fw.SUMMARY_FILE.read_text())["n_flagged"] == 0
