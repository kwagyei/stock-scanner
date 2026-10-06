# Stockholm small-cap attention scanner (v1)

A research and alerting tool. It flags unusual attention on Stockholm small caps and
tracks hypothetical (paper) trades so you can see whether the signals would actually
have made money. **It never places trades.**

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
   exits at +30%, −15%, or after 20 trading days. Each trade is compared with OMXS30
   over the same period.
6. Writes `reports/latest.html` (the digest) and appends to `data/signals_log.csv` and
   `data/paper_trades.csv`. Optionally e-mails the digest.
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

Runs on weekdays at about 08:30 (before the open) and 18:45 (after the close),
Stockholm summer time. That's an hour earlier in winter. Your PC doesn't need to be on.

1. Create a free GitHub account, then a **private** repository (e.g. `stock-scanner`).
2. Upload all files from this folder, keeping the `.github/workflows/scan.yml` path.
   Easiest way: on the repo page, *Add file → Upload files* and drag the whole folder in.
3. Go to the repo's *Actions* tab and enable workflows if prompted.
4. Open *attention-scan → Run workflow* to do a first run manually.
5. Read the result in one of these places:
   - **Job summary:** *Actions → latest run → Summary*. Shows the alerts table.
   - **Full digest:** `reports/latest.html` in the repo, or the *digest* artifact on the run page.
     Download and open it in a browser.
6. Optional e-mail: set `email.enabled: true` in `config.yaml` and add these secrets under
   *Settings → Secrets and variables → Actions*:
   `SMTP_HOST`, `SMTP_PORT` (587), `SMTP_USER`, `SMTP_PASS`, `EMAIL_TO`.
   - For Outlook/Microsoft 365, use host `smtp.office365.com`. SMTP auth must be allowed
     on the account.
   - For Gmail, use host `smtp.gmail.com` with an app password.
   - Using a separate mailbox for this is a good idea.

The workflow commits the updated logs back to the repo after each run, so the history
builds up over time.

## Option B: run it locally on Windows

```powershell
cd path\to\stock-scanner
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scanner.py
start reports\latest.html
```

To schedule it, open Task Scheduler and choose *Create Basic Task* → Weekly → Mon–Fri at 08:30.
Set the action to run `C:\path\to\stock-scanner\.venv\Scripts\python.exe` with
argument `scanner.py`, and set "Start in" to the project folder. Repeat for 18:45.
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
- **Paper fills:** these are idealised. There's no courtage, no spread, and thin small
  caps may not fill at the open price.

*General information and a research tool. Not investment advice.*
