@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting TokenLens...
python -m tokenlens
if errorlevel 1 (
    echo.
    echo [Error] Python 3.10+ is required to run TokenLens.
    pause
)
