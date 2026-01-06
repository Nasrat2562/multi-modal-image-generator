@echo off
title Toon Transformer - Port 5006
cd /d "C:\Users\USER\Documents\AIVScode\style-transfer-app"
call tf_env\Scripts\activate

echo ============================================
echo   STARTING TOON STYLE TRANSFORMER
echo ============================================
echo.
echo Toon Style Transformer starting on port 5006
echo Toon page: http://localhost:5006/toon
echo.
echo The server is starting. This window will stay open.
echo Press Ctrl+C to stop the server when done.
echo ============================================
echo.

python run.py --port 5006 --mode toon