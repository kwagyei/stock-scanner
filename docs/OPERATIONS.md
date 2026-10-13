# Operating the scanner

How to set it up, schedule it, read its output and maintain it. For what the project is and
why it's built this way, see the [README](../README.md).

## What each run does

1. Downloads daily price and volume for every stock in `watchlist.csv` (Yahoo Finance).
2. Downloads the past week of news per company (Google News RSS, Swedish edition) and the
   company's own releases from MFN (mfn.se, which also carries Cision-distributed releases).
3. Scores each stock:

   | Signal | Points |
   |---|---|
   | Company release in last 24h (MFN feed, or a wire source in Google News) | +2 |
   | Volume ≥ 5x its 60-day average (≥ 3x gives +1) | +2 |
   | Price move ≥ ±10% in one day | +1 |
   | News spike: ≥ 3 items in 24h and ≥ 3x the week's daily average | +1 |
   | Market cap < 5 bn SEK (only if another signal fired) | +1 |

4. Flags **hype warnings**: up ≥ 50% in 5 days with no company release found.
5. Updates the **paper-trade log**. When score ≥ 4, it buys at the next day's open and
   exits at +30%, −15%, or after 20 trading days. Each trade is compared with the
   Stockholm small-cap index (primary benchmark) and OMXS30 (secondary) over the same period.
   Three paper books run side by side on the same exits (see [LLM release scoring](#llm-release-scoring)):
   `v0` (these rules), `v0b` and `v1_<prompt>`.
   A **partial-exit variant** is tracked next to it on the same entries (columns `pe_*`):
   sell half at +30%, move the stop on the rest to the entry price, and sell the rest at
   that stop or after 20 trading days. The main rules are not affected; the digest and
   weekly summary show both, so you can compare them.
6. Writes `reports/latest.html` (the digest) and appends to `data/signals_log.csv` and
   `data/paper_trades.csv`. `data/scores_log.csv` gets one row per stock per run, scored
   or not, with each score component in its own column (`pts_release`, `pts_volume`,
   `pts_price`, `pts_news`, `pts_small_cap`) and the raw numbers behind them. Optionally e-mails the digest. It also writes `DASHBOARD.md` and
   Markdown copies of the reports (`reports/*.md`), which GitHub shows as pages.
7. On the Friday evening run (or with `--weekly`), writes a **weekly summary** to
   `reports/weekly_latest.html` and adds a row to `data/weekly_summary.csv`:
   - *Signal hit rate* over the last 12 weeks: for each signal (score ≥ 3, one per stock
     and day), the return 1, 5 and 20 trading days after the signal-day close, and
     whether it beat the index over the same days. Split by score.
   - *Paper P&L*: trades closed this week and all time, win rate, average return vs
     the index, and P&L in SEK at a fixed size per trade (no costs).

   The CSV keeps one row per week, so it becomes the long-run record.

All thresholds are in `config.yaml`.

## Option A: run it in the cloud with GitHub Actions (recommended)

Runs on weekdays at 08:07 (before the open) and 18:52 (after the close), Stockholm time
all year, plus the fund watch every day at 07:13. Your PC doesn't need to be on. The runs
are started by a free external timer, cron-job.org (see [Scheduling](#scheduling-cron-joborg)),
because GitHub's own schedule proved unreliable for this repo. A backup morning run at 08:37
does nothing if the 08:07 run already completed. Both morning runs stay before the 09:00 open,
because new paper trades buy at the next open (a run after the open buys at that day's open).

The work is split over two repositories:

| Repository | Holds | Runs |
|---|---|---|
| `stock-scanner` (public) | Code, tests, docs | CI only (lint + tests on every push) |
| `stock-scanner-data` (private) | Everything the runs write: `data/`, `reports/`, `DASHBOARD.md`, `fund-watch/data/`, `fund-watch/reports/` | The scheduled scan and fund watch |

The data repository's workflows check out the latest code into `code/`, run it with
`SCANNER_STATE_DIR` set to the data repository, and commit the output there. Because the code
repository is public, no token is needed to read it. The deployed workflow files are kept,
versioned with the code, in [`deploy/data-repo/`](../deploy/data-repo/).

**Setting it up from scratch**

1. Create a **private** repository, e.g. `stock-scanner-data`.
2. Copy everything in `deploy/data-repo/` into it (keep the `.github/workflows/` paths).
   If your code repository isn't `kwagyei/stock-scanner`, change `repository:` in both workflows.
3. Go to its *Actions* tab and enable workflows if prompted.
4. Open *attention-scan → Run workflow* to do a first run manually. The first run creates the
   folders and the dashboard.
5. Read the result in one of these places:
   - **Dashboard:** `DASHBOARD.md` in the data repository.
   - **Job summary:** *Actions → latest run → Summary*. Shows the alerts table.
   - **Full digest:** `reports/latest.html`, or the *digest* artifact on the run page.
6. Optional e-mail: set `email.enabled: true` in `config.yaml` and add these secrets under
   *Settings → Secrets and variables → Actions* of the **data** repository:
   `SMTP_HOST`, `SMTP_PORT` (587), `SMTP_USER`, `SMTP_PASS`, `EMAIL_TO`.
   - For Outlook/Microsoft 365, use host `smtp.office365.com`. SMTP auth must be allowed
     on the account.
   - For Gmail, use host `smtp.gmail.com` with an app password.
   - Using a separate mailbox for this is a good idea.

Each run commits its output to the data repository, so the history builds up there, and the
code repository's history stays code only. To run locally instead, leave `SCANNER_STATE_DIR`
unset: the output then goes into the code folder, where it is git-ignored.

### Scheduling (cron-job.org)

GitHub's own `schedule:` never fired for this repo for two days, then fired about 7 hours
late. So the runs are started by cron-job.org (free), which calls GitHub's "Run workflow" API
at fixed times, the same as pressing the button. Runs started this way begin within seconds.
The `schedule:` lines stay in the workflows as a fallback; their late runs skip themselves.

**Setup** (done 8 Oct 2026; repeat if you ever move the repo)

1. **GitHub token.** <https://github.com/settings/personal-access-tokens/new> (fine-grained):
   repository access *Only select repositories → stock-scanner-data*, permission
   *Actions: Read and write*, nothing else. Copy it; GitHub shows it once. It lives only in
   cron-job.org. Never commit it or paste it anywhere else. It can start this repo's
   workflows but not read or change the code.
2. **cron-job.org account.** Set the account time zone to *Europe/Stockholm*.
3. **Four jobs.** All four are `POST` with the same four headers (Advanced tab, one key/value
   row each):

   | Key | Value |
   |---|---|
   | `Accept` | `application/vnd.github+json` |
   | `Authorization` | `Bearer <token>` (one space after Bearer, no quotes) |
   | `X-GitHub-Api-Version` | `2022-11-28` |
   | `Content-Type` | `application/json` |

   URL: `https://api.github.com/repos/kwagyei/stock-scanner-data/actions/workflows/<file>/dispatches`

   | Job | `<file>` | Request body | Schedule (Europe/Stockholm) |
   |---|---|---|---|
   | Fund watch | `fund_watch.yml` | `{"ref":"main","inputs":{"slot":"daily"}}` | every day 07:13 |
   | Morning scan | `scan.yml` | `{"ref":"main","inputs":{"slot":"morning"}}` | Mon–Fri 08:07 |
   | Morning backup | `scan.yml` | `{"ref":"main","inputs":{"slot":"morning"}}` | Mon–Fri 08:37 |
   | Evening scan | `scan.yml` | `{"ref":"main","inputs":{"slot":"evening"}}` | Mon–Fri 18:52 |

   Make one, then use *Duplicate* for the rest and change the title, time, body and (for the
   fund watch) the URL. Turn on "notify me when an execution fails".
4. **Test.** Use *Test run* on each job; it should return **204**. On GitHub, *Actions* shows
   a new run of the right workflow. Test while that slot has already run today (e.g. in the
   evening), and the run skips itself instead of scanning.

**How the slot works.** A run with `slot` skips itself when that slot already has a run today
(Stockholm time): morning = a scan commit since 05:00, evening = since 18:00, daily = a fund
watch commit since midnight. That stops the backup, a retry or a late GitHub run from scanning
twice. A run without `slot` (the *Run workflow* button) always runs. An evening-slot run that
starts before 18:00 skips itself too: today's bar isn't complete until then
(`complete_bar_after`), so it would score yesterday's bar. That matters for GitHub's fallback
`schedule:`, which runs in UTC: its evening run is at 17:07 UTC, i.e. 19:07 in summer and
18:07 in winter. `tests/test_schedule.py` checks the cron times against the bar cutoff in both
seasons.

**When something goes wrong**

| Symptom | Cause / fix |
|---|---|
| cron-job.org shows **401** | Token expired or mistyped: create a new one (step 1) and paste it into the `Authorization` header of all four jobs. Set a reminder before the expiry date. |
| **404** | Wrong URL: check the repo name and `scan.yml` / `fund_watch.yml`. |
| **422** | Request body isn't valid JSON, or `ref` isn't `main`. |
| 204 but no new data | Open the run on GitHub: "Already scanned in this slot; skipping" is normal. Otherwise read the failed step. |
| No call at all | cron-job.org's execution history for the job; the job may be disabled. |

## Option B: run it locally on Windows

```powershell
cd path\to\stock-scanner
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scanner.py
start reports\latest.html
```

To schedule it, open Task Scheduler and choose *Create Basic Task* → Weekly → Mon–Fri at 08:07.
Set the action to run `C:\path\to\stock-scanner\.venv\Scripts\python.exe` with
argument `scanner.py`, and set "Start in" to the project folder. Repeat for 18:52.
The PC must be on at those times.

## First things to do

1. **Verify the tickers.** Search each one on finance.yahoo.com. Any symbol that returns
   no data is listed in red at the top of the digest.
2. **Extend the watchlist.** Either add rows by hand (`source` = `manual`), or rebuild it
   from the exchange lists (see below). Use a distinctive `query` for names that cause
   news noise (e.g. "Elon Group", not "Elon"). If the name is a common word and no
   phrase helps, set `title_match` (e.g. `Stille`): only headlines that contain that word
   (exact case, Swedish genitive -s allowed) are counted as news for that stock.
3. **Leave the rules alone for 2–3 months.** Don't tune thresholds after every run, or you'll
   fit the noise. After about 30 closed paper trades, compare the average return with the
   index column. If the signals don't beat the index on paper, they won't with real money.

## Judging the results (evaluate.py)

`evaluate.py` decides whether the signals are worth trading with real money. It reads the
closed trades in `data/paper_trades.csv`, takes off a round-trip cost, and compares each
trade with the index over the same days. It never changes any data.

```
python evaluate.py                    # main paper-trade rules (the v0 book)
python evaluate.py --strategy v0b     # another book: v0, v0b or v1_<prompt version>
python evaluate.py --compare          # v0 / v0b / v1 side by side, plus the LLM event study
python evaluate.py --variant          # the partial-exit variant
python evaluate.py --cost 0.5 --stake 0.05 --min-trades 50
python evaluate.py --direction drop      # only trades that followed a big drop (or: jump, flat)
```

Defaults (also used for the one-line verdict in the evening digest and on the dashboard):
costs from `costs:` in `config.yaml`, per side, so 2 × (0.1% commission + 0.5% half spread)
= 1.2% per round trip. Where the bid-ask spread was logged at signal time (`spread_pct`),
it replaces the 0.5% half spread. `--cost` sets one round-trip figure for every trade
instead. Trades are judged against the small-cap index where the trade has it, else
OMXS30. 10% of the account per trade (`evaluation.stake`), 30 closed trades before a
verdict (`eval.min_trades`).

**The verdict**

- **NOT YET**: fewer than 30 closed trades. Keep the rules fixed and wait.
- **PASS**: all four hold: enough trades; the average trade beats the index after
  costs; it still does with the single best trade removed; and the t-statistic is above 2.
- **FAIL**: anything else. On paper the signals don't beat simply holding the index.

**Reading the numbers**

| Line | What it tells you |
|---|---|
| Average excess over the index | The edge per trade after costs. This is the number that matters. |
| … without the best trade / best two | Whether the result depends on one or two lucky trades. If it turns negative, it does. |
| Win rate, average win / loss | How it makes money: many small wins, or a few big ones. |
| Longest losing streak | How many losses in a row to expect. Can you sit through that? |
| Worst drawdown | The biggest fall from a high point, if each trade used 10% of the account. |
| t-statistic | How large the edge is compared with its noise. Above 2 means it is unlikely to be chance alone. |
| Chance the true excess is above 0 | The same question asked by resampling the trades 10,000 times. 95%+ is encouraging. |
| By exit reason | Whether targets, stops or time exits drive the result. |
| By signal-day move | The scanner buys after big drops as well as big jumps (a ±10% move both score). This shows whether buying after a **drop**, a **jump** or **no big move** works. Also in the weekly summary (hit rates) and on the Paper trades page. |

A PASS on paper is necessary, not sufficient: paper fills ignore slippage, and thin small
caps rarely fill at the open price. If it passes, start small.

## LLM release scoring

An experiment: does Claude reading the company's own press release add anything to the
rules? Three paper books run in parallel, with the same exits, costs and benchmark.

**Off by default, so it costs nothing.** With `llm.enabled: false` (the default) nothing is
sent to Claude and no API key is needed; v0 and v0b still run (both free) and there is no
v1 book. Set `llm.enabled: true` and add the API key (below) to start v1. The Anthropic API
is billed per use (roughly $3–10 a month here), separately from any Claude subscription.

| Strategy | Score |
|---|---|
| `v0` | The rules above, unchanged. "Company release" = MFN feed or a wire source in Google News. |
| `v0b` | v0, but only primary releases (MFN feed, which carries Cision too) count, timed to the signal bar (below). Since 13 Oct 2026 it also skips two kinds of entry (`paper_trading.v0b_entry_filters`): a stock it was stopped out of within the last 10 weekdays, and a signal day that moved more than ±30%. |
| `v1_<prompt>` | v0b with the flat +2 replaced by LLM points: material and direction +2 → +3; direction +1 → +2; anything else → 0. No buy when a material release scores direction ≤ −2 (`block_buy`). |

**Why v0b has entry filters.** The first week's three closed trades were all −15% stop-losses,
each right after an extreme move in a tiny, volatile stock (Oncopeptides after a +45% day) or a
re-buy the day after a stop (Fastator, twice). From 13 Oct 2026 v0b skips both. v0 keeps no
filters, so it stays the control. v0b's trades signalled before 13 Oct used no filters: read its
results from that date on when judging the filters.

With several releases on one signal bar, the most positive decides the points and the most
negative decides blocking. Routine titles (`release_ignore_titles`) count for nothing and are
not sent to the LLM.

**Timing.** A release published before 09:00 Stockholm on a trading day can be traded at that
day's open; otherwise at the next trading day's open. Trading days are the dates in the price
data, so weekends and holidays need no calendar. The signal bar is the trading day before that
entry day: v0b and v1 count releases from 09:00 on the signal bar up to the next open, and the
paper trade buys at the next open as usual. The event study uses the same entry day.

**As-of time (all strategies, since 13 Oct 2026).** Every bar is scored "as of" the next
trading day's 09:00 open, or the time of the run if that's earlier
(`strategies.as_of_for_bar`). Releases and news published at or after it don't count for
that bar, in v0 as well: v0's 24-hour release and news windows end at the as-of time. Evening
runs (after 18:00) and morning runs (before 09:00) are unaffected; a run during the session,
or one still on an older bar because Yahoo hadn't delivered the latest day, no longer sees
news from after the open its trade would buy at. The next trading day is the first weekday
after the bar that has data or hasn't finished yet, so a holiday's releases still count for
the bar before it. A day counts as traded if the benchmark has it or at least two stocks do,
including rows Yahoo left without a close; one stray row can't turn a holiday into a trading
day. A weekday with no rows anywhere is treated as a holiday, and the scan logs a warning
("treated as a market holiday"); if that day did trade, its data is missing and the bar
before it is scored with that day's news. The `as_of` column in the score log records the
cut-off used for every stock. Before the fix, the two daytime runs on 8 Oct credited three score-3
alerts (Freja, Acconeer, AlzeCure) with releases published after the open; they're in the
signals log and weekly hit rates, but no paper trade came from them.

**Files**

| File | What's in it |
|---|---|
| `news_sources.py` | Fetches each company's MFN feed once per run (v0 parses the same bytes) and stores releases in `data/releases.csv`: primary rows with the full text, plus Google News wire matches as `aggregator` rows (no text). Deduplicated on `release_id` (hash of source + URL) and on near-identical titles per stock within 24h. |
| `llm_score.py` | Sends each new primary release to Claude (`data/llm_scores.csv`). JSON output checked against a fixed schema; one retry on invalid output, then `status=error`. Scored once per `release_id` + prompt version. `api_error` rows (API down, rate limit, no key) are retried next run. `python llm_score.py --backfill` also scores releases before `llm.score_from_date`; those rows get `is_backfill=true` and are never used in trading or evaluation. |
| `prompts/score_v1.md` | The prompt. **Never edit a prompt after it has been used live**: add `score_v2.md`, set `llm.prompt_version: v2` (a new strategy `v1_v2`). On first live use the prompt is fingerprinted in `data/used_prompts.json`; an edited prompt makes scoring stop with an error in the report. |
| `strategies.py` | Timing, v0b/v1 points and blocking. |
| `data/llm_events.csv` | Return after each scored release at 1, 5 and 20 trading days, the small-cap index over the same days, and the abnormal return (difference minus costs). Rebuilt every run. |

`paper_trades.csv` has a `strategy` column (trades from before the experiment are `v0`) plus
`bench_primary_*` and `spread_pct`. `signals_log.csv` and `scores_log.csv` add `score_v0`,
`score_v0b`, `score_v1`, `block_buy`, the release ids, model, prompt version and block threshold.

**Config keys**: `llm.model`, `llm.prompt_version`, `llm.effort` (fixed per prompt version),
`llm.score_from_date` (go-live: only releases published from this date are scored),
`llm.max_calls_per_run` (cost cap), `llm.max_body_chars`, `llm.block_if_direction_lte`,
`llm.skip_routine`, `llm.points.*`; `costs.commission_pct`, `costs.half_spread_pct`;
`eval.min_trades`, `eval.min_events`, `eval.horizons`; `benchmark.primary`, `benchmark.secondary`.

**Adding the API key.** Create a key at console.anthropic.com. In the GitHub repo:
*Settings → Secrets and variables → Actions → New repository secret*, name
`ANTHROPIC_API_KEY`. The workflow passes it to the scan step only; the SDK reads it from the
environment and it is never printed or written to a file. Locally, `export ANTHROPIC_API_KEY=…`
in your terminal before `python scanner.py`. Without a key (and `llm.enabled: true`) the scan still
runs; the report says how many releases were not scored. Expected cost with Sonnet 5.5: about
$0.01–0.03 per release, roughly $3–10 a month, at most 50 calls per run. Set a monthly spend
limit in the console.

**Reading the comparison** (`python evaluate.py --compare`, and the *Evaluation* page). One
column per strategy, each with its own verdict (same four checks as above, against small
caps; OMXS30 shown for reference). Then the event study: abnormal returns grouped by the LLM's
`direction`, `surprise` and `event_type` (N, mean, median, t, hit rate at each horizon) and six
checks at 5 days: enough events (`eval.min_events`, 300); returns rise with direction; "up minus
down" (direction > 0 minus direction < 0) is positive, in both halves of the sample, and without
the best event; t above 2. Everything says **NOT YET** until each strategy has 30 closed trades
and the study has 300 events. The LLM helps only if v1 beats v0b *and* the event study passes;
v0b vs v0 shows what timing and primary-only releases do on their own.

**Data sources**

- Releases: `https://mfn.se/all/a/<mfn_slug>.rss?limit=20` (full text in `<x:content type="text">`).
- Primary benchmark: OMX Stockholm Small Cap PI from Nasdaq Nordic's public chart endpoint,
  `https://api.nasdaq.com/api/nordic/instruments/IX3364/chart?assetClass=INDEXES&fromDate=…&toDate=…`
  (`benchmark.primary: "nasdaq:IX3364"`). Yahoo has no usable history for this index. Like the
  share screener it is undocumented; if it fails, the report warns and the trade's primary
  benchmark stays blank (the evaluation then uses OMXS30 for it).
- Spreads: bid and ask from the Nasdaq share screener at scan time.

## Rebuilding the watchlist

```
python build_watchlist.py --dry-run   # see what would change
python build_watchlist.py             # write watchlist.csv and data/watchlist_review.csv
```

It takes every share on Nasdaq Stockholm Small Cap and First North Stockholm
(about 440), keeps those with a median daily turnover of at least 1 M SEK over the last
60 days (one share class per company), and finds each company's MFN feed by ISIN.
Rows with `source` = `manual` are always kept. Anything that fails a check (new listing,
no Yahoo data, no MFN match) goes to `data/watchlist_review.csv`; add it by hand as a
manual row if you want it. Settings are under `universe:` in `config.yaml`.

Run it by hand about once a month, look at the diff, then commit. Each build writes a new
`# version:` line, and new paper trades are tagged with it (`universe` column), so the
weekly summary can show results per watchlist version. Trades from before versioning are
`v1`.

The lists come from Nasdaq Nordic's public share screener (`api.nasdaq.com`), which
is the data behind nasdaqomxnordic.com, not a documented API. If it changes, the
builder stops with an error and your watchlist is left as it is.

## Known limitations (v1)

- **Prices:** Yahoo data is unofficial and may be delayed. The tool uses daily bars only,
  so moves inside a single day are invisible.
- **Release detection:** primarily the per-company MFN feed (`mfn_slug` in `watchlist.csv`;
  find it in the company's URL on mfn.se). Google News rarely indexes Cision/MFN, so a
  stock with a blank slug will almost never get the release signal. Every MFN item counts,
  including routine ones (buyback updates, share-count notices).
- **macOS local runs:** the python.org installer ships without CA certificates, so news and
  release fetches fail with `CERTIFICATE_VERIFY_FAILED`. Run *Install Certificates.command*
  in `/Applications/Python 3.x/` once.
- **No social media yet.** X requires paid API access, and scraping it breaks its terms.
  Reddit and the Placera forum are candidates for v2.
- **Paper fills:** these are idealised. The evaluation deducts commission and spread, but
  thin small caps may not fill at the open price at all. The logged spread is a snapshot
  at scan time, not the spread at the next open.

*General information and a research tool. Not investment advice.*
