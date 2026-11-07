@echo off
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%PROJECT_PYTHON%" goto run
set "PROJECT_PYTHON=%~dp0..\..\work\venv\Scripts\python.exe"
if exist "%PROJECT_PYTHON%" goto run
set "PROJECT_PYTHON=python"
:run
echo Open http://127.0.0.1:8765 in your browser when the ready message appears.
echo Keep this window open during your demonstration.
"%PROJECT_PYTHON%" -u demo.py
pause
