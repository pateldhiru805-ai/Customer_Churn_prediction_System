@echo off
title Customer Churn Prediction System
echo =========================================================
echo   Starting Customer Churn Prediction System...
echo =========================================================
cd /d "%~dp0"
python run.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    echo Please ensure Python is installed and requirements are satisfied:
    echo pip install -r requirements.txt
    pause
)
