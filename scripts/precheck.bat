@echo off
REM Pre-close tranche heads-up (15 min before US close). See intraday_precheck.py.
setlocal
cd /d "c:\codes\spass mit Zuhalk\edgestack"
set PY=.venv\Scripts\python.exe
if not exist logs mkdir logs
%PY% scripts\intraday_precheck.py >> logs\precheck.log 2>&1
endlocal & exit /b %errorlevel%
