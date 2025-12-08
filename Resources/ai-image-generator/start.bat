@echo off
chcp 65001 > nul
title 🎨 AI Image Generator - Full Version

echo ========================================
echo 🎨 AI Image Generator - Full AI Version
echo ========================================
echo.

set PYTHON_PATH=C:\Users\USER\python-3.11.9-embed-amd64
set PROJECT_PATH=%~dp0
set OUTPUT_PATH=%PROJECT_PATH%output
set MODEL_PATH=C:\Users\USER\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5

REM Check if Python exists
if not exist "%PYTHON_PATH%\python.exe" (
    echo ❌ ERROR: Python not found at:
    echo    %PYTHON_PATH%
    echo.
    echo Please check the Python path
    pause
    exit /b 1
)

echo ✅ Python found: %PYTHON_PATH%
echo 📁 Project folder: %PROJECT_PATH%
echo 📂 Output folder: %OUTPUT_PATH%
echo.

REM Create output folder
if not exist "%OUTPUT_PATH%" mkdir "%OUTPUT_PATH%"
if not exist "%OUTPUT_PATH%\generated_images" mkdir "%OUTPUT_PATH%\generated_images"

REM Check if model already exists
if exist "%MODEL_PATH%" (
    echo ✅ AI Model already downloaded
    echo    Location: %MODEL_PATH%
    echo.
) else (
    echo ⚠️ AI Model not found locally
    echo    Will download on first run (~5GB)
    echo.
)

REM Install required packages
echo 📦 Checking Python packages...
echo.

echo Step 1: Checking/Installing basic packages...
"%PYTHON_PATH%\python.exe" -m pip install --upgrade pip
"%PYTHON_PATH%\python.exe" -m pip install flask pillow numpy --quiet

echo Step 2: Checking/Installing PyTorch...
"%PYTHON_PATH%\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet

echo Step 3: Checking/Installing AI libraries...
"%PYTHON_PATH%\python.exe" -m pip install diffusers transformers accelerate scipy safetensors --quiet

echo ✅ Packages ready!
echo.

REM Run the application
echo ========================================
echo 🚀 Starting AI Image Generator
echo ========================================
echo.
echo 🌐 Open your browser and go to:
echo    http://localhost:5001
echo.
echo 📂 Generated images will be saved to:
echo    %OUTPUT_PATH%\generated_images
echo.
if exist "%MODEL_PATH%" (
    echo ✅ Using cached AI model
    echo    Generation will start immediately
) else (
    echo ⚠️ First time setup:
    echo    - Will download AI model (~5GB)
    echo    - This may take 10-30 minutes
    echo    - Only happens once!
)
echo.
echo 🎨 Features:
echo    • Real AI image generation
echo    • Download button for images
echo    • Organized output folder
echo    • Multiple styles & presets
echo.
echo 🛑 Press Ctrl+C to stop
echo ========================================
echo.

"%PYTHON_PATH%\python.exe" app.py

pause