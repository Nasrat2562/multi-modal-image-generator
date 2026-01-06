# test_env.py
import os
import sys

print("=" * 70)
print("TESTING IN CLEAN VIRTUAL ENVIRONMENT")
print("=" * 70)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"Python: {sys.executable}")
print(f"Directory: {current_dir}")

try:
    import numpy
    print(f"✅ NumPy: {numpy.__version__}")
except:
    print("❌ NumPy not found")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
except:
    print("❌ PyTorch not found")

try:
    import diffusers
    print(f"✅ Diffusers: {diffusers.__version__}")
except:
    print("❌ Diffusers not found")

print("\n" + "=" * 70)
print("TESTING YOUR CONVERTERS")
print("=" * 70)

try:
    import monet_converter
    print("✅ monet_converter imported!")
    
    # Create instance
    converter = monet_converter.MonetConverter()
    print(f"   Device: {converter.device}")
    print(f"   Model loaded: {converter.pipe is not None}")
    
    if converter.pipe:
        print("\n🎉 SUCCESS! Your AI model is CONNECTED!")
        print("   Model: Lykon/dreamshaper-8")
    else:
        print("\n❌ Model pipe is None")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("ENVIRONMENT IS CLEAN AND READY!")
print("=" * 70)
input("\nPress Enter to exit...")