@echo off
echo ============================================
echo   STARTING MONET STYLE TRANSFORMER
echo ============================================
echo.
echo Monet Style Transformer starting on port 5004
echo Monet page: http://localhost:5004/monet
echo.

cd /d "C:\Users\USER\Documents\AIVScode\style-transfer-app"
call tf_env\Scripts\activate.bat

python run.py --port 5004 --mode monet

pause