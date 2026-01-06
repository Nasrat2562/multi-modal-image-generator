@echo off
echo ============================================
echo AI ART TRANSFORMER - QUICK START
echo ============================================
echo.

REM Check if virtual environment exists
if exist ai_env (
    echo Activating virtual environment...
    call ai_env\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv ai_env
    call ai_env\Scripts\activate.bat
    
    echo Installing dependencies...
    pip install --upgrade pip
    pip install Pillow flask werkzeug
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install diffusers transformers accelerate
)

echo.
echo Starting AI Art Transformer...
echo.

cd app
python main_final.py

pause