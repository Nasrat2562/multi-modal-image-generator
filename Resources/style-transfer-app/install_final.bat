@echo off
echo ============================================
echo AI ART TRANSFORMER - FINAL INSTALLATION
echo ============================================
echo.

REM Remove old environment if exists
if exist ai_env (
    echo Removing old environment...
    rmdir /s /q ai_env
)

echo Creating fresh virtual environment...
python -m venv ai_env
call ai_env\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo.
echo ===== INSTALLING BASIC PACKAGES =====
echo Installing pre-built Pillow...
pip install --prefer-binary Pillow==10.0.0

echo Installing Flask and Werkzeug...
pip install flask==2.3.3 werkzeug==2.3.7

echo.
echo ===== INSTALLING AI PACKAGES =====
echo Installing PyTorch (latest CPU version)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo Installing HuggingFace packages...
pip install huggingface-hub==0.20.3
pip install diffusers==0.26.3 transformers==4.38.2 accelerate==0.27.2

echo.
echo ===== VERIFYING INSTALLATION =====
python -c "import PIL; print('✅ Pillow:', PIL.__version__)"
python -c "import flask; print('✅ Flask:', flask.__version__)"
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import diffusers; print('✅ Diffusers:', diffusers.__version__)"

echo.
echo ✅ INSTALLATION COMPLETE!
echo.
pause