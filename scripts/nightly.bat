@echo off
REM Thin launcher: the Python orchestrator is fail-fast and publishes atomically.
setlocal
cd /d "c:\codes\spass mit Zuhalk\edgestack"
set PY=.venv\Scripts\python.exe
if not exist logs mkdir logs
set LOG=logs\nightly_%date:~-4%%date:~-7,2%%date:~-10,2%.log

echo ===== EdgeStack nightly %date% %time% ===== >> "%LOG%" 2>&1
%PY% scripts\nightly.py --config configs\live.yaml >> "%LOG%" 2>&1
set EXIT_CODE=%errorlevel%
REM The Yahoo fetch intermittently dies with a fatal interpreter error
REM (native-extension GIL corruption; crashes observed 2026-07-18/19).
REM The orchestrator is atomic and resumable, so one retry is safe.
REM (goto, not a parenthesized block: %errorlevel% must expand at run time)
if "%EXIT_CODE%"=="0" goto post_nightly
echo ===== nightly exited %EXIT_CODE%; retrying once %time% ===== >> "%LOG%" 2>&1
%PY% scripts\nightly.py --config configs\live.yaml >> "%LOG%" 2>&1
set EXIT_CODE=%errorlevel%
:post_nightly
REM ACN/CTSH tranche-entry watcher; runs even if nightly failed (flags stale data itself).
%PY% scripts\tranche_watch.py >> "%LOG%" 2>&1
REM Forward collectors: PIT universe snapshot, intraday archive, EDGAR earnings refresh.
%PY% scripts\universe_snapshot.py >> "%LOG%" 2>&1
%PY% scripts\intraday_collector.py >> "%LOG%" 2>&1
%PY% scripts\edgar_earnings.py --symbols ACN,CTSH,EPAM,DXC,IBM,IT --refresh-days 7 >> "%LOG%" 2>&1
echo ===== done %time% (exit %EXIT_CODE%) ===== >> "%LOG%" 2>&1
endlocal & exit /b %EXIT_CODE%
