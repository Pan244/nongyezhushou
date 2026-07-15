@echo off
title Agri Assistant

echo ============================================
echo   Agri Plant Protection Assistant v2.0
echo   Backend: http://localhost:8000
echo ============================================

cd /d "%~dp0"

start "" http://localhost:8000
F:\vision\Python312\python.exe app.py
pause
