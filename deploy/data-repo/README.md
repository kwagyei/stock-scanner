# stock-scanner-data (private)

Output of the scheduled runs of [kwagyei/stock-scanner](https://github.com/kwagyei/stock-scanner):
scan logs, paper trades, reports and the fund risk watch.

**📊 [Open the dashboard](DASHBOARD.md)**

The workflows in `.github/workflows/` check out the latest code from the public repository,
run it with `SCANNER_STATE_DIR` pointing here, and commit the results. They are started by an
external timer (cron-job.org); see the code repository's
[docs/OPERATIONS.md](https://github.com/kwagyei/stock-scanner/blob/main/docs/OPERATIONS.md).
Don't edit the workflows here directly: change them in `deploy/data-repo/` in the code
repository and copy them over.
