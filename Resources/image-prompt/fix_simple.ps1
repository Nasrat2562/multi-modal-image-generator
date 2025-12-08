Write-Host "🔧 Fixing package version compatibility..." -ForegroundColor Yellow

# Uninstall problematic versions
pip uninstall diffusers transformers huggingface-hub accelerate -y

# Install known compatible versions
Write-Host "📦 Installing compatible versions..." -ForegroundColor Cyan

pip install diffusers==0.19.3
pip install transformers==4.30.2
pip install huggingface-hub==0.16.4
pip install accelerate==0.20.3
pip install safetensors==0.3.3

Write-Host "✅ Package compatibility fixed!" -ForegroundColor Green

# Test the fix
Write-Host "📝 Testing imports..." -ForegroundColor Yellow
python test_fix.py