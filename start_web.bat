@echo off
title EcoSort AI Web Server
echo ========================================================
echo       Khoi dong EcoSort AI - Web Classifier & Camera
echo ========================================================
echo Dang mo may chu tai http://localhost:8000 ...
start http://localhost:8000
.\.venv\Scripts\python.exe app.py
pause
