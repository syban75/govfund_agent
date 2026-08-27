@echo off

cd /d "%~dp0src"

"%~dp0.venv\Scripts\python.exe" -m streamlit run .\src\streamlit_app.py --server.port 8514 --server.address 0.0.0.0