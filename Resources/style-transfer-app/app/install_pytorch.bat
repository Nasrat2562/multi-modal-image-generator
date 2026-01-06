@echo off
echo ============================================
echo INSTALLING PYTORCH FOR EMBEDDED PYTHON
echo ============================================
echo.

REM Navigate to your embedded Python
cd /d "C:\Users\USER\python-3.11.9-embed-amd64"

echo 1. Updating pip...
python -m pip install --upgrade pip

echo 2. Installing PyTorch (CPU version)...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo 3. Installing Hugging Face libraries...
python -m pip install diffusers transformers accelerate

echo 4. Installing web dependencies...
python -m pip install flask pillow

echo.
echo ============================================
echo VERIFICATION
echo ============================================
echo Testing installation...
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import diffusers; print('✅ Diffusers:', diffusers.__version__)"

echo.
echo ✅ Installation complete!
pause