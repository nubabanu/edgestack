@echo off
REM Watchdog launcher for the read-only API server; used by the auto-start task.
REM Restarts the server if it ever exits, with a short backoff to avoid spin.
setlocal
cd /d "c:\codes\spass mit Zuhalk\edgestack"
if not exist logs mkdir logs

:serve
echo ===== EdgeStack API start %date% %time% ===== >> "logs\api_serve.log" 2>&1
.venv\Scripts\edgestack.exe api serve --host 0.0.0.0 --port 8000 --config configs\live.yaml >> "logs\api_serve.log" 2>&1
echo ===== EdgeStack API exited (%errorlevel%) %date% %time%; restarting in 10s ===== >> "logs\api_serve.log" 2>&1
timeout /t 10 /nobreak > nul
goto serve
