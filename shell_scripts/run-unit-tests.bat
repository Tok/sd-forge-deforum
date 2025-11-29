@echo off
REM run-unit-tests.bat - Run Deforum unit tests (Windows)

setlocal

set SCRIPT_DIR=%~dp0
set EXTENSION_DIR=%SCRIPT_DIR%..\
set FORGE_DIR=%SCRIPT_DIR%..\..\..\

echo ========================================
echo Running Deforum Unit Tests
echo ========================================
echo.
echo Extension directory: %EXTENSION_DIR%
echo Forge directory: %FORGE_DIR%
echo.

cd /d "%EXTENSION_DIR%"

REM Run pytest on unit tests directory using Forge venv
"%FORGE_DIR%\venv\Scripts\python.exe" -m pytest tests/unit/ -v %*

if errorlevel 1 (
    echo.
    echo ========================================
    echo Tests FAILED
    echo ========================================
    exit /b 1
) else (
    echo.
    echo ========================================
    echo All tests PASSED
    echo ========================================
    exit /b 0
)
