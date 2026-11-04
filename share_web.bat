@echo off
title EcoSort AI - Web & Cloudflare Tunnel
cd /d "%~dp0"

echo ========================================================
echo       Khoi dong EcoSort AI Web Server & Public Link
echo ========================================================
echo.

:: 1. Khoi dong app Python trong background
echo [1/2] Dang bat Web Server tai cong 8000...
start "EcoSort Web Server" /min .\.venv\Scripts\python.exe app.py

:: Doi 3 giay de server khoi dong hoan tat
timeout /t 3 /nobreak >nul

:: 2. Chay Cloudflare Tunnel tao link public
echo [2/2] Dang tao duong link public cho nguoi khac...
echo.
echo ========================================================
echo CHU Y: Tim dong co link dang:
echo https://xxxx-xxxx-xxxx.trycloudflare.com
echo Copy link do gui cho nguoi khac de ho truy cap!
echo ========================================================
echo.

.\cloudflared.exe tunnel --url http://localhost:8000

pause
