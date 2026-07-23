@echo off
REM Thin launcher: the Python orchestrator is fail-fast and publishes atomically.
REM Every stage runs through run_stage.py so its outcome lands in
REM artifacts\nightly_status.json; nightly_status_report.py folds them into the
REM aggregate exit code (0 ok / 1 auxiliary failure / 2 core failure) and
REM alerts via Telegram + toast on any failure.
setlocal
cd /d "c:\codes\spass mit Zuhalk\edgestack"
set PY=.venv\Scripts\python.exe
if not exist logs mkdir logs
set LOG=logs\nightly_%date:~-4%%date:~-7,2%%date:~-10,2%.log

echo ===== EdgeStack nightly %date% %time% ===== >> "%LOG%" 2>&1
%PY% scripts\nightly_status.py start >> "%LOG%" 2>&1
%PY% scripts\run_stage.py --stage nightly --core -- scripts\nightly.py --config configs\live.yaml >> "%LOG%" 2>&1
REM The Yahoo fetch intermittently died with a fatal interpreter error
REM (native-extension GIL corruption; crashes observed 2026-07-18/19). Batches
REM are now isolated in subprocesses, but the whole-run retry stays as a
REM belt-and-braces backstop: the orchestrator is atomic and resumable.
REM (goto, not a parenthesized block: %errorlevel% must expand at run time)
if "%errorlevel%"=="0" goto post_nightly
echo ===== nightly exited %errorlevel%; retrying once %time% ===== >> "%LOG%" 2>&1
%PY% scripts\run_stage.py --stage nightly --core -- scripts\nightly.py --config configs\live.yaml >> "%LOG%" 2>&1
:post_nightly
REM ACN/CTSH tranche-entry watcher; runs even if nightly failed (flags stale data itself).
%PY% scripts\run_stage.py --stage tranche_watch -- scripts\tranche_watch.py >> "%LOG%" 2>&1
REM Oil surge watch EOD pass: canonical-close shock/dip state machine (dip tickets study-gated).
%PY% scripts\run_stage.py --stage oil_surge_watch -- scripts\oil_surge_watch.py --eod >> "%LOG%" 2>&1
REM Retired WIND leverage handler: settles only exposure pending at retirement, never opens new.
%PY% scripts\run_stage.py --stage wind_paper_book -- scripts\wind_paper_book.py >> "%LOG%" 2>&1
REM Prospective unlevered paired-fill audit; research ledger only, never orders or alerts.
%PY% scripts\run_stage.py --stage wind_execution_shadow -- scripts\wind_execution_shadow.py >> "%LOG%" 2>&1
REM Forward collectors: PIT universe snapshot, intraday archive, EDGAR earnings refresh.
%PY% scripts\run_stage.py --stage universe_snapshot -- scripts\universe_snapshot.py >> "%LOG%" 2>&1
%PY% scripts\run_stage.py --stage intraday_collector -- scripts\intraday_collector.py >> "%LOG%" 2>&1
%PY% scripts\run_stage.py --stage edgar_earnings -- scripts\edgar_earnings.py --symbols ACN,CTSH,EPAM,DXC,IBM,IT --refresh-days 7 >> "%LOG%" 2>&1
%PY% scripts\nightly_status_report.py >> "%LOG%" 2>&1
set EXIT_CODE=%errorlevel%
echo ===== done %time% (exit %EXIT_CODE%) ===== >> "%LOG%" 2>&1
endlocal & exit /b %EXIT_CODE%
