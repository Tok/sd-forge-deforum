@echo off
REM run-api-tests.bat - Run Deforum API tests (Windows)
REM
REM This script runs the Deforum API test suite. The Forge server must be running.
REM
REM Usage:
REM   run-api-tests.bat              Run tests against running server
REM   run-api-tests.bat --start      Start server, run tests, stop server

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set EXTENSION_DIR=%SCRIPT_DIR%..\
set FORGE_DIR=%SCRIPT_DIR%..\..\..\

echo ========================================
echo Deforum API Tests
echo ========================================
echo.

REM Check if we need to start the server
if "%~1"=="--start" (
    echo Starting Forge server...
    cd /d "%FORGE_DIR%"
    start "Forge Server" /B "%FORGE_DIR%\venv\Scripts\python.exe" webui.py --deforum-api --api --nowebui

    echo Waiting for server to start ^(30 seconds^)...
    timeout /t 30 /nobreak >nul

    set SERVER_STARTED=1
) else (
    echo Running tests against existing server at http://localhost:7860
    set SERVER_STARTED=0
)

echo.
echo Running API tests...
echo.

cd /d "%EXTENSION_DIR%"
"%FORGE_DIR%\venv\Scripts\python.exe" -m pytest tests/ -v -k "not unit" %*

set TEST_RESULT=%ERRORLEVEL%

REM Stop server if we started it
if "%SERVER_STARTED%"=="1" (
    echo.
    echo Stopping Forge server...
    taskkill /F /FI "WINDOWTITLE eq Forge Server*" 2>nul
)

echo.
if %TEST_RESULT% == 0 (
    echo ========================================
    echo All API tests PASSED
    echo ========================================
    exit /b 0
) else (
    echo ========================================
    echo API tests FAILED
    echo ========================================
    exit /b 1
)
