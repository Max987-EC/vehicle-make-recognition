@echo off
setlocal
cd /d "%~dp0"
set "PIP_CACHE_DIR=%CD%\.cache\pip"
set "TMP=%CD%\.cache\tmp"
set "TEMP=%CD%\.cache\tmp"
if not exist "%TMP%" mkdir "%TMP%"
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 (
        python -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))"
        if errorlevel 1 (
            echo Python 3.11 is required. Install it or add its executable to PATH.
            exit /b 1
        )
        python -m venv .venv
        if errorlevel 1 exit /b 1
    )
)
.venv\Scripts\python.exe -c "import sys; sys.exit(sys.version_info[:2] != (3, 11))"
if errorlevel 1 (
    echo Existing .venv is unusable or uses the wrong Python. Move .venv out of this folder and rerun task 01.
    exit /b 1
)
.venv\Scripts\python.exe -m pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b 1
if exist requirements.lock.txt (
    .venv\Scripts\python.exe -m pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
) else (
    .venv\Scripts\python.exe -m pip install -r requirements.txt
)
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -B -X utf8 reproduce.py environment

