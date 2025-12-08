print("🧪 Testing fixed package versions...")

try:
    import diffusers
    print(f"✅ Diffusers: {diffusers.__version__}")
except Exception as e:
    print(f"❌ Diffusers: {e}")

try:
    from huggingface_hub import hf_hub_download
    print("✅ HuggingFace Hub imports work!")
except Exception as e:
    print(f"❌ HuggingFace Hub: {e}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    from sd_xl_converter import SDXLGhibliConverter
    print("✅ SDXL converter imports work!")
except Exception as e:
    print(f"❌ SDXL converter: {e}")

print("🎉 Testing complete!")