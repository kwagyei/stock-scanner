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
- E-mail: set `email.enabled: true` and add the SMTP secrets. By default it sends only when
  something is flagged.

## Maintain
- Update `holdings` from the fund's monthly factsheet.
- Add or remove keywords and topics freely.
- Lower `flag_min_score` for more alerts; raise it for fewer.
