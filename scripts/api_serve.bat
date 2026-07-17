@echo off
REM Thin launcher for the read-only API server; used by the auto-start scheduled task.
setlocal
cd /d "c:\codes\spass mit Zuhalk\edgestack"
if not exist logs mkdir logs
echo ===== EdgeStack API start %date% %time% ===== >> "logs\api_serve.log" 2>&1
.venv\Scripts\edgestack.exe api serve --host 0.0.0.0 --port 8000 --config configs\live.yaml >> "logs\api_serve.log" 2>&1
endlocal
