@echo off

cd /d "%~dp0src"

"%~dp0.venv\Scripts\python.exe" -m uvicorn main:app --reload --port 8014