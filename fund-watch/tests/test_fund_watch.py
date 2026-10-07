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


def fund(cfg, fid="jpm_taiwan"):
    return next(f for f in cfg["funds"] if f["id"] == fid)


def item(title, summary="", hours_ago=1, source="Reuters"):
    return {"title": f"{title} - {source}", "link": "http://x", "source": source,
            "ts": NOW - dt.timedelta(hours=hours_ago), "summary": summary}


def classify(title, cfg, fid="jpm_taiwan", summary="", source="Reuters"):
    f = fund(cfg, fid)
    return fw.classify(item(title, summary, source=source), fw.compile_categories(cfg),
                       fw.build_name_sets(f), cfg, fund=f)


# --------------------------------------------------------------------------- whole-word matching

def _war_cfg(cfg):
    return {**cfg, "risk_categories": {"geopolitics": {"severity": 3, "keywords": ["war", "PLA", "ban"]}}}


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


def test_holding_names_are_whole_words_too(cfg):
    assert classify("Strike vote at TSMCX plant", cfg)["entities"] == []
    assert classify("Strike vote at TSMC's plant", cfg)["entities"] == ["TSMC"]


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
    ("Fed raises rates as recession fears grow", "macro", 2),
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
                                   "War game highlights Taiwan response", "Microsoft fires CEO",
                                   "Google, Constellation strike 20-year nuclear power deal",
                                   "Missile strikes hit base", "Nvidia AI bubble fears overblown",
                                   "Fed rate hike odds recede", "Bond sell-off eases as Nasdaq hits record",
                                   "Amazon crash attorneys say pilot pairing was key flaw"])
def test_tuned_false_positives_stay_quiet(cfg, title):
    for fid in ("jpm_taiwan", "avanza_global"):
        c = classify(title, cfg, fid)
        assert c is None or c["score"] < fw.load_config(fw.ROOT / "fund_watch.yaml")["scoring"]["flag_min_score"] \
            or "geopolitics" in c["categories"], (fid, c and c["categories"])


@pytest.mark.parametrize("title", ["Magnitude 6.4 earthquake strikes off Hualien",
                                   "Powerful quake rocks eastern Taiwan", "Mag. 7.1 quake near Taiwan"])
def test_strong_quakes_still_flag(cfg, title):
    assert "natural_disaster" in classify(title, cfg)["categories"]


def test_publisher_name_is_not_content(cfg):
    # "Simply Wall Street" / "Taipei Times" in the source must not count as Wall Street / Taiwan
    assert classify("Tariff exposed consumer stocks to watch", cfg, "avanza_global",
                    summary="Tariff exposed consumer stocks Simply Wall Street",
                    source="Simply Wall Street")["relevant"] is False
    assert classify("Strike vote at chip plant", cfg, summary="Strike vote Taipei Times",
                    source="Taipei Times")["relevant"] is False


def test_holding_exclude_phrases(cfg):
    assert classify("Smaller apple crop after summer drought", cfg, "avanza_global")["entities"] == []
    assert classify("Apple hit by drought at supplier", cfg, "avanza_global")["entities"] == ["Apple"]
    assert classify("Add Us On Google: local police investigation", cfg, "avanza_global")["entities"] == []


# --------------------------------------------------------------------------- scoring

def test_scoring_bonuses(cfg):
    s = cfg["scoring"]
    base = classify("Strike vote at chip plant", cfg)                                # labour, 2
    assert (base["score"], base["entities"], base["relevant"]) == (2, [], False)
    tw = classify("Strike vote at Taiwan chip plant", cfg)
    assert tw["score"] == 2 + s["region_bonus"] and tw["relevant"]
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
    assert full["score"] == 2 + s["holding_bonus"] + s["dependency_bonus"] + s["region_bonus"]


def test_fund_severity_overrides(cfg):
    # The index fund rates single-company legal news LOW and macro HIGH.
    tw = classify("Google faces antitrust probe", cfg, "jpm_taiwan")
    gl = classify("Google faces antitrust probe", cfg, "avanza_global")
    assert (tw["severity"], gl["severity"]) == (2, 1)
    assert classify("Fed raises rates as recession looms", cfg, "avanza_global")["severity"] == 3
    assert classify("Fed raises rates as recession looms", cfg, "jpm_taiwan")["severity"] == 2


def test_relevance_gate_moves_unrelated_high_severity_to_watch(cfg):
    far = classify("Strong earthquake hits Peru", cfg)                                # HIGH, score 3, no link
    near = classify("Strong earthquake hits Taiwan", cfg)
    assert far["score"] == 3 and not far["relevant"]
    flagged, watch = fw.split_flagged([far, near], cfg)
    assert flagged == [near] and watch == [far]
    off = {**cfg, "scoring": {**cfg["scoring"], "require_relevance": False}}
    assert fw.split_flagged([far, near], off)[0] == [far, near]


def test_fund_flag_threshold(cfg):
    g = fund(cfg, "avanza_global")
    three = classify("Wall Street selloff deepens", cfg, "avanza_global")            # market 2 + region 1
    four = classify("Nvidia plunges in sharp selloff", cfg, "avanza_global")         # market 2 + top 2
    assert (three["score"], four["score"]) == (3, 4)
    assert fw.split_flagged([three, four], cfg, g) == ([four], [three])


def test_fund_name_counts_as_relevant(cfg):
    assert classify("JPM Taiwan Fund manager change announced", cfg)["relevant"]
    assert classify("Avanza Global hit by sell-off", cfg, "avanza_global")["relevant"]


@pytest.mark.parametrize("title", [
    "Fed unexpectedly raises rates",                         # macro HIGH + Fed
    "Swedish krona surges against the dollar",               # macro HIGH + krona
    "Recession fears grip Wall Street",
    "US slaps tariffs on chips, Nvidia tumbles",             # trade HIGH + top holding
    "Apple cuts orders for iPhone parts",                    # demand + top holding
    "Microsoft issues profit warning",
    "Global stocks plunge as China moves to blockade Taiwan",
])
def test_global_fund_flags_systemic_and_top_holding_events(cfg, title):
    c = classify(title, cfg, "avanza_global")
    assert fw.split_flagged([c], cfg, fund(cfg, "avanza_global"))[0] == [c], (c["score"], c["categories"])


@pytest.mark.parametrize("title", ["Google faces Polish antitrust probe", "Micron union authorizes strike",
                                   "Tesla recalls 2 million cars", "Ford selloff tests investors"])
def test_global_fund_keeps_routine_company_news_on_watch(cfg, title):
    c = classify(title, cfg, "avanza_global")
    assert c is None or fw.split_flagged([c], cfg, fund(cfg, "avanza_global"))[0] == []


# --------------------------------------------------------------------------- queries

def test_queries_cover_both_funds_without_duplicates(cfg):
    qs = fw.build_queries(cfg)
    assert len(qs) == len(set(qs))
    owners = fw.query_funds(cfg)
    assert owners['"Avanza Global"'] == {"avanza_global"}
    assert owners['"TSMC" OR "Taiwan Semiconductor"'] == {"jpm_taiwan"}       # global relies on it
    assert owners['"Apple"'] == {"avanza_global"}
    assert any(q.startswith('"Apple" (TSMC') for q in owners)                 # Taiwan's customer search


# --------------------------------------------------------------------------- dedup

def test_item_key_ignores_source_suffix_case_and_punctuation():
    a = fw.item_key("TSMC halts production after quake - Reuters")
    assert a == fw.item_key("TSMC Halts Production After Quake! - Bloomberg.com")
    assert a != fw.item_key("TSMC resumes production after quake - Reuters")


# --------------------------------------------------------------------------- end to end

class _FrozenDatetime(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz else NOW.replace(tzinfo=None)


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


def _log():
    import csv
    if not fw.LOG_FILE.exists():
        return []
    return list(csv.DictReader(fw.LOG_FILE.open(encoding="utf-8")))


def test_dedup_across_queries_and_seen_store(monkeypatch, cfg):
    quake = item("Strong earthquake halts TSMC fabs in Taiwan")
    dup = item("Strong earthquake halts TSMC fabs in Taiwan", source="Bloomberg")   # same story, other outlet
    calm = item("TSMC opens visitor centre")
    feeds = {'"TSMC"': [quake, calm], "Taiwan earthquake": [dup]}
    calls = run(monkeypatch, feeds)
    assert len(calls) == len(fw.build_queries(cfg))                                  # every search ran
    rows = _log()
    assert sorted(r["fund"] for r in rows) == ["avanza_global", "jpm_taiwan"]       # scored for both funds
    assert len(json.loads(fw.SEEN_FILE.read_text())) == 2                           # unrelated item marked seen

    run(monkeypatch, feeds)                                                           # same news again
    assert len(_log()) == 2
    assert "Nothing new above the flag threshold." in (fw.REPORT_DIR / "fund_watch_latest.html").read_text()


def test_article_scored_for_both_funds_with_their_own_rules(monkeypatch):
    feeds = {'"TSMC"': [item("Strong earthquake halts TSMC fabs in Taiwan")]}
    run(monkeypatch, feeds)
    by_fund = {r["fund"]: r for r in _log()}
    assert by_fund["jpm_taiwan"]["score"] == str(3 + 2 + 1)        # HIGH + top holding + Taiwan
    assert by_fund["jpm_taiwan"]["entities"] == "TSMC"
    assert by_fund["avanza_global"]["score"] == str(3 + 1)         # HIGH + dependency (TSMC)


def test_other_funds_broad_searches_dont_leak_through_dependencies(monkeypatch):
    # Google is a Taiwan-fund customer, but plain Google news comes from the global fund's search.
    feeds = {'"Alphabet" OR "Google"': [item("Google faces antitrust probe in Poland")],
             '"Google" (TSMC': [item("Google cuts orders for TSMC chips")]}
    run(monkeypatch, feeds)
    rows = {(r["fund"], r["title"].split(" - ")[0]) for r in _log()}
    assert ("jpm_taiwan", "Google faces antitrust probe in Poland") not in rows
    assert ("avanza_global", "Google faces antitrust probe in Poland") in rows
    assert ("jpm_taiwan", "Google cuts orders for TSMC chips") in rows


def test_all_flag_ignores_and_keeps_seen_store_and_log(monkeypatch):
    feeds = {'"TSMC"': [item("Strong earthquake halts TSMC fabs in Taiwan")]}
    run(monkeypatch, feeds)
    before = fw.SEEN_FILE.read_text()
    run(monkeypatch, feeds, "--all")
    assert "Strong earthquake halts TSMC" in (fw.REPORT_DIR / "fund_watch_latest.html").read_text()
    assert fw.SEEN_FILE.read_text() == before and len(_log()) == 2


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
    feeds = {'"TSMC"': [item("TSMC halts production in Taiwan, Nvidia hit"),   # 2+2+1+1 = 6
                        item("Strike vote at Quanta plant"),                   # 2+1 = 3
                        item("Strong earthquake hits Peru")]}                  # 3, not relevant -> watch
    run(monkeypatch, feeds)
    tw = [r for r in _log() if r["fund"] == "jpm_taiwan"]
    assert [r["score"] for r in tw] == ["6", "3", "3"]
    html = (fw.REPORT_DIR / "fund_watch_latest.html").read_text()
    taiwan = html.split("<h2>")[1]
    flagged, watch = taiwan.split("<h3>Watch")
    assert "Peru" in watch and "Peru" not in flagged and "Quanta" in flagged


def test_old_single_fund_log_gets_fund_column(monkeypatch):
    fw.DATA_DIR.mkdir()
    fw.LOG_FILE.write_text("run_utc,published_utc,score,severity,categories,entities,source,title,link\n"
                           "2026-10-07T12:49+00:00,,3,MEDIUM,labour,,Reuters,Old item,http://o\n")
    run(monkeypatch, {'"TSMC"': [item("Strong earthquake halts TSMC fabs in Taiwan")]})
    rows = _log()
    assert list(rows[0]) == fw.LOG_COLUMNS
    assert (rows[0]["fund"], rows[0]["title"]) == ("jpm_taiwan", "Old item")
    assert {r["fund"] for r in rows[1:]} == {"jpm_taiwan", "avanza_global"}


def test_feed_failure_warns(monkeypatch, capsys, cfg):
    failed = fw.feedparser.FeedParserDict(bozo=1, bozo_exception="SSL: CERTIFICATE_VERIFY_FAILED", entries=[])
    monkeypatch.setattr(fw.feedparser, "parse", lambda url, agent=None: failed)
    assert fw.fetch_feed('"TSMC"', cfg) == []
    assert "feed failed" in capsys.readouterr().err


# --------------------------------------------------------------------------- page + summary

def test_markdown_page(monkeypatch):
    feeds = {'"TSMC"': [item("TSMC | halts production in Taiwan, Nvidia hit"),
                        item("China blockade of Taiwan begins"),
                        item("Strong earthquake hits Peru")]}
    run(monkeypatch, feeds)
    md = (fw.REPORT_DIR / "fund_watch_latest.md").read_text()
    assert md.startswith("# 🛡️ Fund risk watch") and "(../../DASHBOARD.md)" in md
    assert "| 🇹🇼 **Taiwan** | **2** | **1** | 1 |" in md        # stories, high, watch (no grouping here)
    assert "| 🌍 **Global** |" in md
    assert "## 🇹🇼 JPMorgan Funds - Taiwan Fund A (acc) USD" in md and "## 🌍 Avanza Global" in md
    taiwan = md.split("## 🇹🇼")[1].split("## 🌍")[0]
    assert "> [!WARNING]\n> **1 high-severity item(s)**" in taiwan
    assert "TSMC \\| halts production" in taiwan and "🔴&nbsp;HIGH" in taiwan and "🟠&nbsp;MEDIUM" in taiwan
    flagged, watch = taiwan.split("<details>")
    assert "Peru" in watch and "Peru" not in flagged
    assert md.count("<details>") == md.count("</details>")


def test_summary_written_on_normal_runs_only(monkeypatch):
    feeds = {"Taiwan Strait": [item("China blockade of Taiwan begins")]}
    run(monkeypatch, feeds, "--all")
    assert not fw.SUMMARY_FILE.exists()
    run(monkeypatch, feeds)
    s = json.loads(fw.SUMMARY_FILE.read_text())
    tw = {f["id"]: f for f in s["funds"]}["jpm_taiwan"]
    assert s["run_utc"] == "2026-10-07T05:00+00:00" and [f["id"] for f in s["funds"]] == ["jpm_taiwan", "avanza_global"]
    assert (tw["n_flagged"], tw["n_high"], tw["n_watch"], tw["short"], tw["icon"]) == (1, 1, 0, "Taiwan", "🇹🇼")
    assert tw["top"][0]["severity"] == "HIGH" and tw["top"][0]["title"].startswith("China blockade")
    run(monkeypatch, feeds)                                                        # nothing new next time
    assert all(f["n_flagged"] == 0 for f in json.loads(fw.SUMMARY_FILE.read_text())["funds"])


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


def test_email_only_when_flagged(monkeypatch, cfg):
    sent = []
    monkeypatch.setattr(fw, "send_email", lambda html, subject, required=False: sent.append(subject))
    on = {**cfg, "email": {"enabled": True, "send_when": "flagged"}}
    monkeypatch.setattr(fw, "load_config", lambda path: on)
    monkeypatch.setattr(fw, "fetch_feed", lambda q, c: [])
    monkeypatch.setattr(fw.dt, "datetime", _FrozenDatetime)
    fw.main([])
    assert sent == []                                                            # nothing flagged: no mail
    monkeypatch.setattr(fw, "fetch_feed",
                        lambda q, c: [item("China blockade of Taiwan begins")] if "Strait" in q else [])
    fw.main([])
    assert sent == ["Fund risk watch 07 Oct: Taiwan 1 flagged (1 high) · Global 0 flagged (0 high)"]
    fw.main(["--no-email"])
    assert len(sent) == 1


def test_test_email_sends_even_with_nothing_flagged(monkeypatch):
    sent = []
    monkeypatch.setattr(fw, "send_email", lambda html, subject, required=False: sent.append((subject, required)))
    monkeypatch.setattr(fw, "fetch_feed", lambda q, c: [])
    monkeypatch.setattr(fw.dt, "datetime", _FrozenDatetime)
    fw.main(["--test-email"])
    assert sent == [("[TEST] Fund risk watch 07 Oct: Taiwan 0 flagged (0 high) · Global 0 flagged (0 high)", True)]


def test_shipped_config(cfg):
    assert cfg["email"] == {"enabled": True, "send_when": "flagged"}
    assert [f["id"] for f in cfg["funds"]] == ["jpm_taiwan", "avanza_global"]
    g = fund(cfg, "avanza_global")
    assert g["isin"] == "SE0011527613" and g["flag_min_score"] == 4


# --------------------------------------------------------------------------- story grouping

def _stories(cfg, titles, fid="avanza_global", threshold=None):
    """Classify headlines for a fund and group them, like a run does."""
    f = fund(cfg, fid)
    cats, names = fw.compile_categories(cfg), fw.build_name_sets(f)
    items = [item(t, hours_ago=n) for n, t in enumerate(titles, 1)]
    cl = [c for it in items if (c := fw.classify(it, cats, names, cfg, fund=f))]
    cl.sort(key=lambda x: (x["score"], x["severity"]), reverse=True)
    thr = cfg["news"]["story_similarity"] if threshold is None else threshold
    return fw.group_stories(cl, fw.word_weights(items + [item(t) for t in BACKGROUND]), thr)


# Unrelated headlines so word weights look like a real run (common words, a few rare ones).
BACKGROUND = ["Taiwan stocks close higher on tech gains", "Taiwan shares edge lower", "China Taiwan tensions",
              "Stock market today: Nasdaq rises", "Stocks fall as yields climb", "Apple unveils new iPhone",
              "Nvidia earnings preview", "TSMC sales rise in September", "Taiwan dollar firms",
              "Wall Street opens flat", "Fed officials speak this week", "Oil prices steady"] * 3

# Real headline styles from one strike story (many outlets, similar wording).
MICRON = ["Micron Taiwan union wins strike vote over bonus dispute",
          "Micron's Taiwan union wins strike vote with 99% support",
          "Micron Taiwan union wins mandate to strike over bonus dispute",
          "Micron Taiwan workers vote to strike over bonus dispute",
          "Micron Taiwan union authorizes strike over bonus dispute"]


def test_same_story_headlines_are_grouped(cfg):
    stories = _stories(cfg, MICRON, "jpm_taiwan")
    assert len(stories) == 1 and len(stories[0]["similar"]) == 4


def test_very_differently_worded_headlines_may_stay_apart(cfg):
    # Deliberately cautious: wrongly merging two stories would hide one, so headlines that share
    # little wording stay separate even when they are about the same event.
    assert len(_stories(cfg, ["Micron shares fall as Taiwan workers authorize strike amid bonus dispute",
                              "Micron's Taiwan Union Wins Strike Mandate Over Profit-Sharing Demand"],
                        "jpm_taiwan")) == 2


def test_lead_is_the_best_ranked_headline(cfg):
    # the TSMC headline scores highest for the Taiwan fund and leads; the others follow it
    stories = _stories(cfg, ["Taiwan workers vote to strike at TSMC plant over bonus dispute",
                             "Workers vote to strike at chip plant over bonus dispute in Taiwan",
                             "Chip plant workers in Taiwan vote to strike over bonus dispute"], "jpm_taiwan")
    assert len(stories) == 1 and "TSMC" in stories[0]["title"] and stories[0]["entities"] == ["TSMC"]


@pytest.mark.parametrize("a,b", [
    ("Quanta says September server dip is normal as revenue falls 23% from August",
     "EMC stock gains 1.06 percent as August revenue falls"),                       # different companies
    ("McDonald's Faces Antitrust Lawsuit Over AI Pricing",
     "Expedia hit with antitrust class action over algorithm"),
    ("Micron Taiwan plant faces strike over bonus dispute",
     "Micron Taiwan plant faces lawsuit over bonus dispute"),                       # same words, other risk
])
def test_different_stories_stay_apart(cfg, a, b):
    assert len(_stories(cfg, [a, b], "jpm_taiwan")) == 2


def test_grouping_can_be_switched_off(cfg):
    assert len(_stories(cfg, MICRON, "jpm_taiwan", threshold=1.01)) == len(MICRON)


def test_grouped_story_end_to_end(monkeypatch):
    feeds = {"Taiwan strike workers": [item(t, source=f"Outlet {n}") for n, t in enumerate(MICRON)]}
    run(monkeypatch, feeds)
    md = (fw.REPORT_DIR / "fund_watch_latest.md").read_text()
    taiwan = md.split("## 🇹🇼")[1].split("## 🌍")[0]
    flagged = taiwan.split("👀 Watch")[0]
    assert flagged.count("](<http://x>)<br><sub>Outlet") == 1                      # one row for the story
    assert flagged.count("\n- [Micron") == 4                                       # 4 in the grouped list
    assert "➕ 4 similar: Outlet 1, Outlet 2, Outlet 3, Outlet 4" in flagged
    assert "📰 Similar headlines grouped into these stories (4)" in flagged
    assert "| 🇹🇼 **Taiwan** | **1**<br><sub>5 articles</sub> |" in md
    assert md.count("<details>") == md.count("</details>")
    rows = [r for r in _log() if r["fund"] == "jpm_taiwan"]
    assert len(rows) == 5 and len({r["story"] for r in rows}) == 1                   # every article logged
    s = {f["id"]: f for f in json.loads(fw.SUMMARY_FILE.read_text())["funds"]}["jpm_taiwan"]
    assert (s["n_flagged"], s["n_flagged_articles"], s["top"][0]["similar"]) == (1, 5, 4)
    html = (fw.REPORT_DIR / "fund_watch_latest.html").read_text()
    assert "+4 similar (Outlet 1, Outlet 2, Outlet 3, Outlet 4)" in html and "Flagged · 1 story" in html


def test_watch_level_headline_joins_a_flagged_story(cfg):
    # without "Taiwan" the second headline alone would be on watch (labour 2 < 3); grouped, it
    # travels with the flagged lead instead of appearing separately.
    stories = _stories(cfg, ["Micron Taiwan union authorizes strike over bonus dispute",
                             "Micron union authorizes strike over bonus dispute"], "jpm_taiwan")
    flagged, watch = fw.split_flagged(stories, cfg, fund(cfg, "jpm_taiwan"))
    assert len(flagged) == 1 and len(flagged[0]["similar"]) == 1 and watch == []


def test_log_with_fund_but_no_story_column_is_upgraded(monkeypatch):
    fw.DATA_DIR.mkdir()
    fw.LOG_FILE.write_text("run_utc,fund,published_utc,score,severity,categories,entities,source,title,link\n"
                           "2026-10-07T13:35+00:00,avanza_global,,4,HIGH,macro,,Forbes,Fed raises rates,http://f\n")
    run(monkeypatch, {})
    rows = _log()
    assert list(rows[0]) == fw.LOG_COLUMNS and (rows[0]["fund"], rows[0]["story"]) == ("avanza_global", "")
