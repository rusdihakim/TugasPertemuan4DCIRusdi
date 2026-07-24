@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.port 8502 --server.headless true --browser.gatherUsageStats false
