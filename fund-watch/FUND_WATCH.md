# Fund risk watch

Daily scan of Google News for anything that could hurt the funds in `fund_watch.yaml`:

| | Fund | ISIN | What drives its risk |
|---|---|---|---|
| 🇹🇼 | JPMorgan Funds - Taiwan Fund A (acc) USD | LU0210528419 | Taiwan, its chip makers and their customers |
| 🌍 | Avanza Global | SE0011527613 | The market as a whole, US big tech, rates, the krona |

Articles are fetched once (about 65 searches) and scored separately for each fund, so one
article can be flagged for both.

## What it searches
**Taiwan fund (about 45 searches)**
- **The fund itself:** manager changes, suspension, mergers.
- **Holdings:** MediaTek, TSMC, Delta, Elite Material, CTBC, Hon Hai, Quanta, Accton, ASE and the Taiwanese banks.
- **Customers:** Nvidia, Apple, AMD, Broadcom, Qualcomm, and the big cloud companies (only together with TSMC, Taiwan, chips or orders).
- **Suppliers:** ASML, Applied Materials, Tokyo Electron, Lam Research.
- **Topics:** the Taiwan Strait, blockade and military activity; earthquakes, typhoons, power and water; strikes;
  export controls, tariffs and chip bans; the TAIEX and the Taiwan dollar; AI spending, chip demand and inventory.

**Avanza Global (about 20 searches)**
An index fund of about 1,000 developed-market stocks, priced in SEK and not currency-hedged.
The top 10 make up about 30%.
- **The fund itself.**
- **Top holdings:** Apple, Nvidia, Microsoft, Alphabet, Amazon (each 3.5–5.6%), plus Broadcom, Meta, Micron, Tesla.
- **Dependency:** TSMC, which makes the chips for several of them. Not searched separately: the Taiwan fund's searches fetch TSMC news.
- **Topics:** Fed and interest rates, US recession, broad sell-offs, the AI bubble and big-tech concentration,
  big-tech antitrust, US tariffs, the Swedish krona against the dollar, global stocks.

A fund only looks at articles from its own searches, plus articles from the other fund's searches
that name its holdings, relevance words or the dependencies it doesn't search for itself. So the
global fund's plain Google news doesn't reach the Taiwan fund just because Google is a customer.

## How it flags
Each new article is matched against 12 risk categories:

| Category | Severity | Avanza Global |
|---|---|---|
| Geopolitics | HIGH | |
| Trade restrictions | HIGH | |
| Natural disaster | HIGH | |
| Fund-specific | HIGH | |
| Macro (recession, rate rises, krona/dollar moves) | MEDIUM | **HIGH** |
| Infrastructure | MEDIUM | |
| Demand and customers | MEDIUM | |
| Company results | MEDIUM | |
| Labour | MEDIUM | LOW |
| Legal and regulatory | MEDIUM | LOW |
| Cyber and operations | MEDIUM | LOW |
| Market stress | LOW | MEDIUM |

The index fund has its own severities (`severity:` in its config), because single-company legal
or labour news rarely moves a 1,000-stock fund, while rates and the krona do. Articles with no
risk words are dropped. The publisher's name is ignored, so "Taipei Times" or "Simply Wall Street"
don't count as Taiwan or Wall Street. Some phrases are excluded (`exclude:`), such as
"strike a deal", "plane crash", "sell-off eases" or "apple crop".

Score = severity (3/2/1), plus these bonuses:
- +2 if a top holding is named (weight ≥ 3%; +1 if another watched holding is named)
- +1 if a customer or supplier is named
- +1 if one of the fund's relevance words is named (Taiwan, Taipei, TAIEX; or Wall Street, S&P 500,
  Nasdaq, Fed, krona and similar)

**Same story, many outlets.** Headlines about the same story are grouped into one item, led by
the best-scoring headline ("➕ 18 similar: TipRanks, Nikkei Asia…"; the others are listed under
the table). Two headlines are grouped when their distinctive words overlap enough
(`news.story_similarity`, 0.5) and they share a risk category, so a company's lawsuit and its
strike stay separate. It's cautious on purpose: very differently worded headlines about the
same event can still show up separately, but two different stories shouldn't be merged.
Counts on the page and the dashboard are stories; the log keeps every article with a `story` id.

**Stories are remembered for 7 days** (`news.story_memory_days`, in `data/fund_watch_stories.json`).
When a later run finds new headlines about a remembered story, it isn't flagged again: it goes
under **🔁 Still developing** ("first seen Tue 6 Oct"), and no e-mail is sent for it. If it got
worse, i.e. it scores or rates higher than before, it is flagged again and marked
**⬆️ escalated**, so an escalation is never hidden. Matching uses the same word overlap and
shared category as the grouping. The first run without a memory file starts from the articles
in the log of the last 7 days.

**Flagged** = score ≥ 3 (≥ 4 for Avanza Global) and the article is tied to the fund: a relevance
word, the fund, a holding or a dependency (`require_relevance`). Everything else goes under
**Watch**. Only articles not seen in earlier runs are shown. It's keyword-based, so expect some
false positives, and read the source.

## Run
- Locally, from the `fund-watch/` folder: `python fund_watch.py` (requires
  `pip install feedparser PyYAML`). Report: `reports/fund_watch_latest.html`.
  `--all` shows everything in the news window without touching the seen-store or the log.
- GitHub: `.github/workflows/fund_watch.yml` (in the repository root) runs every day at
  05:13 UTC, after the tests in `fund-watch/tests`. The *Run workflow* button on the
  **fund-risk-watch** workflow triggers a manual run.
- E-mail: on (`email.enabled: true`), and it sends only when something is flagged. It needs
  five repository secrets (see below). The stock scanner uses the same secrets, but its own
  e-mail stays off (`email.enabled: false` in the root `config.yaml`).

## On the dashboard
Each run also writes `reports/fund_watch_latest.md` (a page GitHub can show) and
`data/fund_watch_latest.json` (a short summary). The stock scanner's dashboard reads that summary
on its own runs: a **Fund watch** button on every page, a count in the at-a-glance row, and the
top flagged items with a link to the full page. It warns if the fund watch hasn't run for 36 hours.

## E-mail setup (Gmail)
1. Use a Gmail account as the sender. A separate account is safest; your own works too
   (it can send to itself).
2. In that account, turn on 2-Step Verification, then create an **app password**
   (Google Account → Security → App passwords). Copy the 16-character password.
   An app password gives full access to that mailbox: revoke it there if the repository
   ever becomes public or gets other collaborators.
3. Add the repository secrets (*Settings → Secrets and variables → Actions*, or the `gh` CLI):

   | Secret | Value |
   |---|---|
   | `SMTP_HOST` | `smtp.gmail.com` |
   | `SMTP_PORT` | `587` |
   | `SMTP_USER` | the new Gmail address (it is also the sender) |
   | `SMTP_PASS` | the app password (not the account password) |
   | `EMAIL_TO` | the address that should receive the alerts |

4. Test: *Actions → fund-risk-watch → Run workflow*, tick **test_email**. You get the current
   report even if nothing is flagged, and the run fails with a clear message if a secret is
   missing or Gmail rejects the login. Check the spam folder the first time.

If an e-mail fails during a normal run, the run fails and the seen-store is not saved, so the
next run reports the same items again (GitHub e-mails you about the failed run).

## Maintain
- Each fund is one entry under `funds:` in `fund_watch.yaml`. To add a fund, copy an entry and
  change the id, names, ISIN, holdings, dependencies, relevance words and topics.
- Update `holdings` from the fund's monthly factsheet.
- Add or remove keywords and topics freely.
- Lower `flag_min_score` for more alerts; raise it for fewer.
