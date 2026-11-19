@echo off
REM One-time setup on a new Windows PC. Needs Python 3.11 or 3.12 + internet.
cd /d "%~dp0"
py -3.12 -m venv .venv || py -3.11 -m venv .venv || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup done. Next:
echo   1. Put dataset\^<Name^>\ folders here
echo   2. run_enroll.bat
pause
