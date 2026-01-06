now modify @echo off
echo ========================================
echo LAUNCHING ALL AI TOOLS - WINDOWS
echo ========================================
echo.
title AI Tools Launcher
set "ROOT=C:\Users\USER\Documents\AIVScode"

echo 1. Starting Age Verification System...
cd /d "%ROOT%\age-predictor"
echo Checking Python for Age Predictor...
if exist "%ROOT%\age-predictor\age_venv\Scripts\python.exe" (
    start "Age Verification" cmd /k "cd /d "%ROOT%\age-predictor\app" && call ..\age_venv\Scripts\activate && echo Starting on http://localhost:5005 && python run.py"
) else (
    echo WARNING: age_venv not found. Using embedded Python...
    start "Age Verification" cmd /k "cd /d "%ROOT%\age-predictor\app" && echo Starting on http://localhost:5005 && "C:\Users\USER\python-3.11.9-embed-amd64\python.exe" run.py"
)
timeout /t 3 /nobreak >nul

echo 2. Starting AI Image Generator...
if exist "%ROOT%\ai-image-generator" (
    start "AI Image Generator" cmd /k "cd /d "%ROOT%\ai-image-generator" && echo Starting on http://localhost:5001 && python app.py"
) else (
    echo WARNING: ai-image-generator folder not found
)
timeout /t 3 /nobreak >nul

echo 3. Starting Ghibli WebApp...
if exist "%ROOT%\ghibli-webapp\ghibli_final" (
    start "Ghibli WebApp" cmd /k "cd /d "%ROOT%\ghibli-webapp" && call ghibli_final\Scripts\activate && echo Starting on http://localhost:5000 && python app.py"
) else (
    echo WARNING: ghibli_final env not found. Trying without...
    start "Ghibli WebApp" cmd /k "cd /d "%ROOT%\ghibli-webapp" && echo Starting on http://localhost:5000 && python app.py"
)
timeout /t 3 /nobreak >nul

echo 4. Starting SDXL Generator...
if exist "%ROOT%\image-prompt\sdxl_env" (
    start "SDXL Generator" cmd /k "cd /d "%ROOT%\image-prompt" && call sdxl_env\Scripts\activate && echo Starting on http://localhost:5003 && python app.py"
) else (
    echo WARNING: sdxl_env not found. Trying without...
    start "SDXL Generator" cmd /k "cd /d "%ROOT%\image-prompt" && echo Starting on http://localhost:5003 && python app.py"
)
timeout /t 3 /nobreak >nul

echo 5. Starting BOTH Monet AND Toon Transformer (using ai_env)...
if exist "%ROOT%\style-transfer-app\ai_env" (
    echo ✅ Found ai_env virtual environment!
    
    REM Start BOTH Monet and Toon on single port
    start "AI Art Transformer" cmd /k "cd /d "%ROOT%\style-transfer-app" && call ai_env\Scripts\activate && cd app && python main_final.py"
    
    echo ✅ AI Art Transformer started: http://localhost:5004
    
) else (
    echo ❌ ai_env not found! Creating...
    
    cd /d "%ROOT%\style-transfer-app"
    python -m venv ai_env
    
    if exist "ai_env" (
        call ai_env\Scripts\activate.bat
        
        pip install --upgrade pip
        pip install Pillow flask werkzeug
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install diffusers transformers accelerate
        
        start "AI Art Transformer" cmd /k "cd /d "%ROOT%\style-transfer-app" && call ai_env\Scripts\activate && cd app && python main_final.py"
        
    ) else (
        echo ❌ Failed to create ai_env
        start "AI Art Transformer" cmd /k "cd /d "%ROOT%\style-transfer-app\app" && python main_final.py"
    )
)


echo.
echo ========================================
echo LAUNCH COMPLETE!
echo ========================================
echo.
echo Check the opened windows for any errors.
echo.
echo URLs to access:
echo.
echo If servers started successfully:
echo ✅ Age Verification:     http://localhost:5005
echo ✅ AI Image Generator:   http://localhost:5001
echo ✅ Ghibli WebApp:        http://localhost:5000
echo ✅ SDXL Generator:       http://localhost:5003
echo ✅ AI Art Transformer:   http://localhost:5004
echo     - Monet Style:       http://localhost:5004/monet
echo     - Toon Style:        http://localhost:5004/toon
echo     - Status:            http://localhost:5004/status
echo.
echo Press any key to exit this window...
pause >nul
