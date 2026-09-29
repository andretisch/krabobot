@echo off
chcp 65001 >nul
cd /d C:\Users\andrey.tishkin\CursorProjects\krabobot
set PYTHONUTF8=1
".\.venv\Scripts\python.exe" -u -m krabobot_voice > ".voice-client.log" 2>&1
