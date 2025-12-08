@echo off
chcp 65001 > nul
echo ========================================
echo 🎨 Full AI Image Generator Installation
echo ========================================

set PYTHON_PATH=C:\Users\USER\python-3.11.9-embed-amd64

if not exist "%PYTHON_PATH%\python.exe" (
    echo ❌ Python not found at: %PYTHON_PATH%
    pause
    exit /b 1
)

echo ⚠️ WARNING: This will install heavy AI dependencies
echo ⚠️ This may take 30+ minutes and requires 10GB+ disk space
echo.
choice /c YN /m "Continue with full installation? (Y/N)"
if errorlevel 2 goto cancel

echo.
echo 📦 Installing full AI dependencies...
echo This will take a while...

"%PYTHON_PATH%\python.exe" -m pip install --upgrade pip

REM Install PyTorch CPU version (lighter)
echo Installing PyTorch...
"%PYTHON_PATH%\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install other dependencies
echo Installing AI libraries...
"%PYTHON_PATH%\python.exe" -m pip install diffusers transformers accelerate scipy safetensors

echo.
echo ✅ Full installation complete!
echo.
echo To use the full AI version, you need to:
echo 1. Update app.py to import AI libraries
echo 2. Restart the application
echo.
echo Current version runs in DEMO mode.
pause
exit

:cancel
echo.
echo Installation cancelled.
echo Running in demo mode is recommended for testing.
pause