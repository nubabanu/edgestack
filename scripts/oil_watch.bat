@echo off
REM Oil geopolitical-shock intraday poll (every 30 min). See oil_surge_watch.py.
setlocal
cd /d "c:\codes\spass mit Zuhalk\edgestack"
set PY=.venv\Scripts\python.exe
if not exist logs mkdir logs
%PY% scripts\oil_surge_watch.py --intraday >> logs\oil_surge_watch.log 2>&1
endlocal & exit /b %errorlevel%
