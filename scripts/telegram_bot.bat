@echo off
REM Watchdog launcher for the read-only Telegram command bot. 30s backoff so a
REM crash-loop cannot hammer the Telegram API. See telegram_bot.py.
setlocal
cd /d "c:\codes\spass mit Zuhalk\edgestack"
if not exist logs mkdir logs
:serve
echo ===== EdgeStack Telegram bot start %date% %time% ===== >> "logs\telegram_bot.log" 2>&1
.venv\Scripts\python.exe -u scripts\telegram_bot.py >> "logs\telegram_bot.log" 2>&1
echo ===== bot exited (%errorlevel%) %date% %time%; restarting in 30s ===== >> "logs\telegram_bot.log" 2>&1
timeout /t 30 /nobreak > nul
goto serve
