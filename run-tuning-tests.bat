@echo off
setlocal enabledelayedexpansion

REM Deforum Parameter Tuning Test Runner (Windows)
REM Runs GPU-required tuning tests against running Forge instance

REM Directories
set SCRIPT_DIR=%~dp0
set FORGE_DIR=%SCRIPT_DIR%..\..
set EXT_DIR=%SCRIPT_DIR%

echo ========================================
echo Deforum Parameter Tuning Test Runner
echo ========================================
echo Forge directory: %FORGE_DIR%
echo Extension directory: %EXT_DIR%
echo ========================================

REM Parse arguments
set START_SERVER=false
set REUSE_SERVER=false
set PYTEST_ARGS=

:parse_args
if "%~1"=="" goto args_done
if "%~1"=="--start-server" (
    set START_SERVER=true
) else if "%~1"=="--reuse-server" (
    set REUSE_SERVER=true
) else (
    set PYTEST_ARGS=!PYTEST_ARGS! %~1
)
shift
goto parse_args

:args_done

REM Start server if requested
if "%START_SERVER%"=="true" (
    echo Starting Forge server with Deforum API...
    echo Server log: %EXT_DIR%test-server.log

    cd /d "%FORGE_DIR%"
    start /B "" "%FORGE_DIR%\venv\Scripts\python.exe" webui.py --skip-prepare-environment --deforum-api --listen > "%EXT_DIR%test-server.log" 2>&1

    echo Server started
    echo Waiting for server to start (max 300s)...

    REM Wait for server (60 attempts * 5 seconds = 300s)
    set /a count=0
    :wait_loop
    set /a count+=1
    if %count% gtr 60 (
        echo [ERROR] Server failed to start in 300s
        exit /b 1
    )

    curl -s http://localhost:7860/deforum_api/jobs/ >nul 2>&1
    if %errorlevel% equ 0 (
        echo [OK] Server is ready!
        goto server_ready
    )

    echo   Still waiting... (%count%s)
    timeout /t 5 /nobreak >nul
    goto wait_loop
)

:server_ready

REM Check dependencies
echo Checking test dependencies...
"%FORGE_DIR%\venv\Scripts\python.exe" -c "import sys; import cv2, PIL, skimage, numpy; print('[OK] All tuning test dependencies available')"
if %errorlevel% neq 0 (
    echo [ERROR] Missing dependencies
    echo Install with: %FORGE_DIR%\venv\Scripts\pip.exe install opencv-python scikit-image
    if "%START_SERVER%"=="true" taskkill /F /IM python.exe /T >nul 2>&1
    exit /b 1
)
echo [OK] Test dependencies OK

REM Run tuning tests
echo ========================================
echo Running Parameter Tuning Tests
echo [WARNING] These tests are SLOW and GPU-intensive
echo [WARNING] Each parameter combination generates multiple images
echo ========================================

cd /d "%EXT_DIR%"

REM If no specific test specified, run all tuning tests
if "%PYTEST_ARGS%"=="" set PYTEST_ARGS=tests/tuning/

"%FORGE_DIR%\venv\Scripts\python.exe" -m pytest !PYTEST_ARGS! -v --tb=short --no-cov

set TEST_EXIT_CODE=%errorlevel%

REM Cleanup
if "%START_SERVER%"=="true" (
    echo Cleaning up...
    echo Stopping Forge server...
    taskkill /F /IM python.exe /T >nul 2>&1
)

REM Report results
if %TEST_EXIT_CODE% equ 0 (
    echo [OK] Tests completed successfully
    echo Results saved to: outputs\deforum-tuning\
) else (
    echo [ERROR] Tests failed or were interrupted
    if "%START_SERVER%"=="true" echo Server log available at: %EXT_DIR%test-server.log
)

exit /b %TEST_EXIT_CODE%
