@echo off
REM EdgeStack nightly pipeline — runs after the US close (scheduled 22:20 local).
REM Chain: prices -> features -> signal report -> paper session -> board ->
REM        picks -> master signal -> app seed -> drift monitor.
setlocal
cd /d "c:\codes\spass mit Zuhalk\edgestack"
set PY=.venv\Scripts\python.exe
if not exist logs mkdir logs
set LOG=logs\nightly_%date:~-4%%date:~-7,2%%date:~-10,2%.log

echo ===== EdgeStack nightly %date% %time% ===== >> "%LOG%" 2>&1
%PY% scripts\live_update.py                              >> "%LOG%" 2>&1
%PY% -m edgestack.cli features build  -c configs\live.yaml >> "%LOG%" 2>&1
%PY% -m edgestack.cli signals generate -c configs\live.yaml >> "%LOG%" 2>&1
%PY% -m edgestack.cli paper run        -c configs\live.yaml >> "%LOG%" 2>&1
%PY% scripts\live_signals.py                             >> "%LOG%" 2>&1
%PY% scripts\make_picks.py                               >> "%LOG%" 2>&1
%PY% scripts\master_signal.py                            >> "%LOG%" 2>&1
%PY% scripts\export_mobile_bundle.py                     >> "%LOG%" 2>&1
%PY% -m edgestack.cli monitor run      -c configs\live.yaml >> "%LOG%" 2>&1
echo ===== done %time% (exit %errorlevel%) ===== >> "%LOG%" 2>&1
endlocal
