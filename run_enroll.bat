@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python enroll.py
echo.
echo Done. Copy embeddings.npz back to the attendance laptop (same folder as config.py).
pause
