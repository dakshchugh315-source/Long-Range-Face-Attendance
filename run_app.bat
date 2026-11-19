@echo off
REM Double-click to start the Face Attendance app (opens in your browser).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python app.py
pause
