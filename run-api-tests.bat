@echo off
REM run-api-tests.bat - Start Forge server with Deforum API and run integration tests
REM
REM Usage:
REM   run-api-tests.bat                          REM Run all tests (force restarts server by default)
REM   run-api-tests.bat --quick                  REM Skip post-processing tests (faster)
REM   run-api-tests.bat --reuse-server           REM Reuse existing server if running
REM   run-api-tests.bat --force-restart          REM Force restart server (default behavior)
REM   run-api-tests.bat --snapshot-update        REM Update test snapshots (after intentional changes)
REM   run-api-tests.bat tests\integration\api_test.py::test_simple_settings  REM Run specific test

setlocal enabledelayedexpansion

REM Configuration
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"
cd ..\..
set FORGE_DIR=%CD%
cd /d "%SCRIPT_DIR%"
set SERVER_URL=http://localhost:7860
set DEFORUM_API_URL=%SERVER_URL%/deforum_api/jobs
set MAX_WAIT=300
set SERVER_LOG=%SCRIPT_DIR%test-server.log
set PID_FILE=%SCRIPT_DIR%.test-server.pid

REM Parse arguments
set QUICK_MODE=false
set FORCE_RESTART=true
set REUSE_SERVER=false
set SNAPSHOT_UPDATE=false
set TEST_ARGS=
set PYTEST_ARGS=

:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="--quick" (
    set QUICK_MODE=true
) else if "%~1"=="--reuse-server" (
    set REUSE_SERVER=true
    set FORCE_RESTART=false
) else if "%~1"=="--force-restart" (
    set FORCE_RESTART=true
    set REUSE_SERVER=false
) else if "%~1"=="--snapshot-update" (
    set SNAPSHOT_UPDATE=true
    set PYTEST_ARGS=!PYTEST_ARGS! --snapshot-update
) else (
    set TEST_ARGS=!TEST_ARGS! %~1
)
shift
goto :parse_args

:args_done

REM Set default test path if none specified
if "!TEST_ARGS!"=="" (
    if "!QUICK_MODE!"=="true" (
        set TEST_ARGS=tests\integration\api_test.py
        echo Quick mode: Running only API tests (skipping post-processing tests)
    ) else (
        set TEST_ARGS=tests\integration\
        echo Running all integration tests (API + post-processing)
    )
)

REM Check if we're in the extension directory
if not exist "scripts\deforum.py" (
    echo [ERROR] Must run from sd-forge-deforum extension directory
    echo Current directory: %CD%
    exit /b 1
)

REM Check if venv exists
if not exist "%FORGE_DIR%\venv" (
    echo [ERROR] Forge venv not found at %FORGE_DIR%\venv
    echo Please run this from the Forge installation
    exit /b 1
)

echo ========================================
echo Deforum Test Runner
echo ========================================
echo Forge directory: %FORGE_DIR%
echo Extension directory: %SCRIPT_DIR%
echo Test arguments: !TEST_ARGS!
echo ========================================
echo.

REM Check if server is already running
curl -s -f "%DEFORUM_API_URL%" >nul 2>&1
set SERVER_RUNNING=%errorlevel%

set EXISTING_SERVER=false
if %SERVER_RUNNING% equ 0 (
    if "!REUSE_SERVER!"=="true" (
        echo [WARNING] Forge server is already running at %SERVER_URL%
        echo Reusing existing server. Press Ctrl+C within 5s to cancel...
        timeout /t 5 /nobreak >nul
        set EXISTING_SERVER=true
    ) else if "!FORCE_RESTART!"=="true" (
        echo [WARNING] Forge server is already running at %SERVER_URL%
        echo Force restart enabled - killing existing server...

        REM Kill existing server processes
        taskkill /F /FI "WINDOWTITLE eq webui.py*" >nul 2>&1
        taskkill /F /FI "IMAGENAME eq python.exe" /FI "COMMANDLINE eq *webui.py*deforum-api*" >nul 2>&1

        REM Wait for server to shut down
        timeout /t 3 /nobreak >nul

        REM Verify server is down
        curl -s -f "%DEFORUM_API_URL%" >nul 2>&1
        if not errorlevel 1 (
            echo [ERROR] Failed to kill existing server - please kill manually
            exit /b 1
        )
        echo [OK] Existing server killed
    )
)

if "!EXISTING_SERVER!"=="false" (
    REM Start Forge server in background
    echo Starting Forge server with Deforum API...
    echo Server log: %SERVER_LOG%
    echo.

    cd /d "%FORGE_DIR%"

    REM Start server in background
    start /B "" "%FORGE_DIR%\venv\Scripts\python.exe" webui.py --skip-prepare-environment --api --deforum-api --skip-version-check --no-gradio-queue > "%SERVER_LOG%" 2>&1

    REM Store PID (note: Windows makes this difficult, we'll use a marker file)
    echo started > "%PID_FILE%"

    echo Server started

    cd /d "%SCRIPT_DIR%"

    REM Wait for server to be ready
    echo Waiting for server to start (max %MAX_WAIT%s)...
    set /a waited=0
    set /a interval=2

    :wait_loop
    curl -s -f "%DEFORUM_API_URL%" >nul 2>&1
    if not errorlevel 1 (
        echo [OK] Server is ready!
        goto :server_ready
    )

    REM Show progress every 10 seconds
    set /a mod=waited %% 10
    if !mod! equ 0 (
        echo   Still waiting... (!waited!s)
    )

    timeout /t %interval% /nobreak >nul
    set /a waited+=interval

    if !waited! lss %MAX_WAIT% goto :wait_loop

    echo [ERROR] Server failed to start after %MAX_WAIT%s
    echo Check server log at: %SERVER_LOG%
    goto :cleanup_and_exit_error

    :server_ready
)

REM Check test dependencies
echo.
echo Checking test dependencies...
"%FORGE_DIR%\venv\Scripts\python.exe" -c "import pytest, syrupy, moviepy" 2>nul
if errorlevel 1 (
    echo Installing test dependencies...
    "%FORGE_DIR%\venv\Scripts\pip.exe" install -q -r requirements-dev.txt
    echo [OK] Test dependencies installed
) else (
    echo [OK] Test dependencies OK
)

REM Run tests
echo.
echo ========================================
if "!SNAPSHOT_UPDATE!"=="true" (
    echo [WARNING] SNAPSHOT UPDATE MODE ENABLED
    echo This will update test snapshots with current output
    echo Only use this after intentional changes!
    echo ========================================
) else (
    echo Running Integration Tests (Unit tests excluded)
    echo ========================================
)
echo.

REM Run tests (don't exit on failure)
"%FORGE_DIR%\venv\Scripts\python.exe" -m pytest !TEST_ARGS! !PYTEST_ARGS! -v --tb=short --no-cov --ignore=tests\unit
set TEST_EXIT_CODE=%errorlevel%

REM Show reminder after snapshot update
if "!SNAPSHOT_UPDATE!"=="true" (
    if %TEST_EXIT_CODE% equ 0 (
        echo.
        echo [OK] Snapshots updated successfully
        echo [WARNING] IMPORTANT: Review changes before committing!
        echo.
        echo Next steps:
        echo   1. Review: git diff tests/__snapshots__/
        echo   2. Commit: git add tests/__snapshots__/ ^&^& git commit -m "test: Update snapshots for [reason]"
        echo   3. Or revert: git checkout tests/__snapshots__/
    )
)

REM Cleanup
:cleanup
echo.
echo Cleaning up...

if "!EXISTING_SERVER!"=="false" (
    if exist "%PID_FILE%" (
        echo Stopping Forge server...
        taskkill /F /FI "WINDOWTITLE eq webui.py*" >nul 2>&1
        taskkill /F /FI "IMAGENAME eq python.exe" /FI "COMMANDLINE eq *webui.py*deforum-api*" >nul 2>&1
        del "%PID_FILE%" >nul 2>&1
    )
)

if %TEST_EXIT_CODE% equ 0 (
    echo [OK] Tests completed successfully
) else (
    echo [FAILED] Tests failed or were interrupted
    echo Server log available at: %SERVER_LOG%
)

exit /b %TEST_EXIT_CODE%

:cleanup_and_exit_error
if exist "%PID_FILE%" (
    taskkill /F /FI "WINDOWTITLE eq webui.py*" >nul 2>&1
    taskkill /F /FI "IMAGENAME eq python.exe" /FI "COMMANDLINE eq *webui.py*deforum-api*" >nul 2>&1
    del "%PID_FILE%" >nul 2>&1
)
exit /b 1
