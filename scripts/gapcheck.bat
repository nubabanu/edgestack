@echo off
REM Pre-open gap-entry check (5 min before the US open). See intraday_precheck.py --pre-open.
setlocal
cd /d "c:\codes\spass mit Zuhalk\edgestack"
set PY=.venv\Scripts\python.exe
if not exist logs mkdir logs
%PY% scripts\intraday_precheck.py --pre-open >> logs\precheck.log 2>&1
endlocal & exit /b %errorlevel%
