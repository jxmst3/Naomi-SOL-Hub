@echo off
REM ============================================================================
REM NAOMI SOL HUB - Windows Launcher
REM ============================================================================

echo ================================================================================
echo                    NAOMI SOL HUB - LAUNCHER
echo                           Version 4.0
echo ================================================================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo [INFO] Virtual environment not found. Creating...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo [INFO] Make sure Python 3.10+ is installed
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
    echo.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if packages are installed
echo [INFO] Checking dependencies...
python -c "import numpy" 2>nul
if errorlevel 1 (
    echo [INFO] Installing required packages...
    echo [INFO] This may take a few minutes...
    pip install numpy scipy
    if errorlevel 1 (
        echo [ERROR] Failed to install packages
        pause
        exit /b 1
    )
    echo [OK] Packages installed
    echo.
)

REM Display menu
:menu
echo.
echo ================================================================================
echo                           NAOMI SOL HUB MENU
echo ================================================================================
echo.
echo   1. Run System (Virtual Mode)
echo   2. Run Tests
echo   3. Generate CAD Files
echo   4. Install Additional Packages
echo   5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto run
if "%choice%"=="2" goto test
if "%choice%"=="3" goto cad
if "%choice%"=="4" goto install
if "%choice%"=="5" goto end

echo [ERROR] Invalid choice. Please enter 1-5.
goto menu

:run
echo.
echo [INFO] Starting Naomi SOL Hub in virtual mode...
echo [INFO] Press Ctrl+C to stop
echo.
python main.py
goto menu

:test
echo.
echo [INFO] Running system tests...
python main.py --mode test
echo.
pause
goto menu

:cad
echo.
echo [INFO] Generating CAD files...
python main.py --mode generate-cad
echo.
echo [INFO] Check output/cad_models/ folder for STL files
pause
goto menu

:install
echo.
echo [INFO] Installing additional packages...
echo [INFO] This includes visualization and analysis tools
pip install pygame matplotlib pandas
echo.
echo [OK] Additional packages installed
pause
goto menu

:end
echo.
echo [INFO] Thank you for using Naomi SOL Hub!
echo.
pause
