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
echo ===== done %time% (exit %EXIT_CODE%) ===== >> "%LOG%" 2>&1
endlocal & exit /b %EXIT_CODE%
