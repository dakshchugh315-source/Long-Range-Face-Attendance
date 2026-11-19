@echo off
REM Usage: run_attend.bat                         (camera from config.py)
REM        run_attend.bat http://192.168.1.5:8080/video   (phone camera)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if "%~1"=="" (python attend.py) else (python attend.py --source "%~1")
pause
