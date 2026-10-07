# Fund risk watch

Daily scan of Google News for anything that could hurt the fund in `fund_watch.yaml`
(currently JPMorgan Funds - Taiwan Fund A (acc) USD, LU0210528419).

## What it searches (about 40 searches per run)
- **The fund itself:** manager changes, suspension, mergers.
- **Holdings:** MediaTek, TSMC, Delta, Elite Material, CTBC, Hon Hai, Quanta, Accton, ASE and the Taiwanese banks.
- **Customers:** Nvidia, Apple, AMD, Broadcom, Qualcomm, and the big cloud companies.
- **Suppliers:** ASML, Applied Materials, Tokyo Electron, Lam Research.
- **Topics:**
  - Taiwan Strait, military activity, blockade
  - earthquakes, typhoons, power and water problems
  - strikes
  - export controls, tariffs, chip bans
  - the TAIEX and the Taiwan dollar
  - AI spending and chip demand
  - semiconductor inventory

## How it flags
Each new article is matched against 11 risk categories:

| Category | Severity |
|---|---|
| Geopolitics | HIGH |
| Trade restrictions | HIGH |
| Natural disaster | HIGH |
| Fund-specific | HIGH |
| Infrastructure | MEDIUM |
| Labour | MEDIUM |
| Demand and customers | MEDIUM |
| Company results | MEDIUM |
| Legal and regulatory | MEDIUM |
| Cyber and operations | MEDIUM |
| Market stress | LOW |

Articles with no risk words are dropped.

Score = severity (3/2/1), plus these bonuses:
- +2 if a top-5 holding is named (+1 if another watched holding is named)
- +1 if a customer or supplier is named
- +1 if Taiwan is mentioned

Score ≥ 3 goes under **Flagged**, but only if the article names Taiwan, the fund, a holding,
a customer or a supplier (`require_relevance`). Everything else goes under **Watch**. Only articles not seen in
earlier runs are shown. It's keyword-based, so expect some false positives, and read the source.

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
- Update `holdings` from the fund's monthly factsheet.
- Add or remove keywords and topics freely.
- Lower `flag_min_score` for more alerts; raise it for fewer.
