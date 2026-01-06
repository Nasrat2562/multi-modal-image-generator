# model_loader.py - COMPLETE MODEL LOADER WITH ALL METHODS
import os
import sys
import subprocess
import importlib
import gc
import time
import traceback

print("\n" + "="*60)
print("🤖 AI MODEL LOADER INITIALIZED")
print("="*60)

class ModelLoader:
    def __init__(self):
        self.device = "cpu"
        self.verbose = True
        print("✅ ModelLoader created")
    
    def install_dependencies(self):
        """Install required packages"""
        print("\n📦 CHECKING/INSTALLING DEPENDENCIES...")
        print("-"*40)
        
        try:
            # List of required packages
            required_packages = [
                ("torch", "torch"),
                ("torchvision", "torchvision"),
                ("diffusers", "diffusers"),
                ("transformers", "transformers"),
                ("accelerate", "accelerate"),
                ("PIL", "Pillow")
            ]
            
            installed_count = 0
            total_count = len(required_packages)
            
            for import_name, package_name in required_packages:
                try:
                    __import__(import_name)
                    print(f"✅ {package_name} already installed")
                    installed_count += 1
                except ImportError:
                    print(f"📥 Installing {package_name}...")
                    try:
                        # Use pip to install
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", 
                            "--quiet", package_name
                        ])
                        print(f"✅ Installed {package_name}")
                        installed_count += 1
                    except Exception as install_error:
                        print(f"❌ Failed to install {package_name}: {install_error}")
            
            print(f"\n📊 Installation Summary: {installed_count}/{total_count} packages ready")
            return installed_count == total_count
            
        except Exception as e:
            print(f"⚠️ Error in install_dependencies: {e}")
            traceback.print_exc()
            return False
    
    def clean_package_cache(self):
        """Clear package cache and force garbage collection"""
        print("🧹 CLEANING PACKAGE CACHE...")
        
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear specific modules from sys.modules
            modules_to_clear = [
                'diffusers', 'transformers', 'torchvision', 'torch',
                'safetensors', 'huggingface_hub', 'accelerate',
                'tensorboard', 'tensorflow', 'protobuf'
            ]
            
            cleared_count = 0
            for module_name in modules_to_clear:
                if module_name in sys.modules:
                    try:
                        del sys.modules[module_name]
                        cleared_count += 1
                        if self.verbose:
                            print(f"   Cleared: {module_name}")
                    except:
                        pass
            
            print(f"✅ Cleared {cleared_count} modules from cache")
            
            # Extra cleanup
            import torch
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error cleaning cache: {e}")
            return False
    
    def check_huggingface_token(self):
        """Check if HuggingFace token is available"""
        try:
            from huggingface_hub import HfFolder
            
            token = HfFolder.get_token()
            if token:
                print("✅ HuggingFace token found")
                return True
            else:
                print("⚠️ No HuggingFace token found (public models will still work)")
                return False
                
        except Exception as e:
            print(f"⚠️ HuggingFace token check failed: {e}")
            return False
    
    def setup_model_paths(self):
        """Setup model cache paths"""
        try:
            # Create cache directory
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Set environment variables
            os.environ['HF_HOME'] = cache_dir
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            os.environ['DIFFUSERS_CACHE'] = cache_dir
            
            print(f"✅ Model cache path: {cache_dir}")
            return cache_dir
            
        except Exception as e:
            print(f"⚠️ Error setting up model paths: {e}")
            return None
    
    def load_model_with_fallback(self, model_name="Lykon/dreamshaper-8"):
        """Load model with multiple fallback strategies"""
        print(f"\n🚀 LOADING MODEL: {model_name}")
        print("-"*40)
        
        # Setup paths first
        self.setup_model_paths()
        
        # Try multiple loading strategies
        strategies = [
            self._load_with_torch_direct,
            self._load_with_diffusers,
            self._load_simple_pipeline
        ]
        
        for i, strategy in enumerate(strategies):
            print(f"\n🔄 Strategy {i+1}/{len(strategies)}...")
            try:
                model = strategy(model_name)
                if model is not None:
                    print(f"✅ Model loaded successfully with strategy {i+1}")
                    return model
            except Exception as e:
                print(f"❌ Strategy {i+1} failed: {e}")
                time.sleep(1)  # Small delay between attempts
        
        print("❌ All loading strategies failed")
        return None
    
    def _load_with_torch_direct(self, model_name):
        """Strategy 1: Direct torch loading"""
        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline
            
            print("   Using: PyTorch + diffusers")
            
            # Set device
            if torch.cuda.is_available():
                self.device = "cuda"
                torch.cuda.empty_cache()
            else:
                self.device = "cpu"
            
            print(f"   Device: {self.device}")
            print(f"   Model: {model_name}")
            print("   Loading...")
            
            # Load the pipeline
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Move to device
            pipe = pipe.to(self.device)
            
            # Enable CPU offload if on CPU
            if self.device == "cpu":
                pipe.enable_attention_slicing()
            
            return pipe
            
        except Exception as e:
            print(f"   Direct load failed: {e}")
            return None
    
    def _load_with_diffusers(self, model_name):
        """Strategy 2: Using diffusers with fallback"""
        try:
            # Try to import with error handling
            import importlib.util
            
            # Check if diffusers is available
            spec = importlib.util.find_spec("diffusers")
            if spec is None:
                print("   diffusers not found")
                return None
            
            import torch
            import diffusers
            
            print("   Using: diffusers pipeline")
            
            # Simple pipeline loading
            pipe = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Set device
            if torch.cuda.is_available():
                self.device = "cuda"
                pipe = pipe.to("cuda")
            else:
                self.device = "cpu"
                pipe = pipe.to("cpu")
                pipe.enable_attention_slicing()
            
            print(f"   Device: {self.device}")
            return pipe
            
        except Exception as e:
            print(f"   Diffusers load failed: {e}")
            return None
    
    def _load_simple_pipeline(self, model_name):
        """Strategy 3: Minimal pipeline loading"""
        try:
            print("   Using: Minimal pipeline")
            
            # Import only what we need
            import torch
            
            # Import diffusers components directly
            from diffusers import StableDiffusionPipeline
            
            # Load with minimal options
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                safety_checker=None
            )
            
            # Set to CPU to save memory
            self.device = "cpu"
            pipe = pipe.to("cpu")
            pipe.enable_attention_slicing()
            
            print(f"   Device: {self.device} (minimal mode)")
            return pipe
            
        except Exception as e:
            print(f"   Minimal load failed: {e}")
            return None
    
    def test_model(self, pipe):
        """Test if model works"""
        if pipe is None:
            print("❌ No model to test")
            return False
        
        try:
            print("\n🧪 TESTING MODEL...")
            
            # Import PIL
            from PIL import Image
            
            # Create a test image
            test_img = Image.new('RGB', (100, 100), color='blue')
            
            # Test with a simple prompt
            test_prompt = "a simple test"
            
            # Try to generate (with minimal settings)
            result = pipe(
                prompt=test_prompt,
                image=test_img,
                strength=0.1,
                guidance_scale=1.0,
                num_inference_steps=1
            )
            
            print("✅ Model test PASSED")
            return True
            
        except Exception as e:
            print(f"❌ Model test FAILED: {e}")
            return False
    
    def get_model_info(self, pipe):
        """Get information about loaded model"""
        if pipe is None:
            return {"status": "no_model"}
        
        info = {
            "status": "loaded",
            "device": self.device,
            "has_cuda": False,
            "model_type": str(type(pipe)).split("'")[1].split(".")[-1],
            "components": []
        }
        
        try:
            import torch
            info["has_cuda"] = torch.cuda.is_available()
            info["torch_version"] = torch.__version__
        except:
            pass
        
        # Check what components are available
        if hasattr(pipe, "components"):
            info["components"] = list(pipe.components.keys())
        
        return info
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 CLEANING UP RESOURCES...")
        
        try:
            import torch
            if hasattr(torch, 'cuda'):
                torch.cuda.empty_cache()
            
            self.clean_package_cache()
            
            print("✅ Cleanup complete")
            return True
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
            return False


# Create global instance for easy import
loader = ModelLoader()

if __name__ == "__main__":
    print("\n🔧 TESTING MODEL LOADER...")
    
    # Test installation
    loader.install_dependencies()
    
    # Clean cache
    loader.clean_package_cache()
    
    # Load a model
    model = loader.load_model_with_fallback("Lykon/dreamshaper-8")
    
    if model:
        print("\n🎉 MODEL LOADER TEST SUCCESSFUL!")
        
        # Get model info
        info = loader.get_model_info(model)
        print(f"\n📊 MODEL INFO:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    else:
        print("\n❌ MODEL LOADER TEST FAILED")
    
    print("\n" + "="*60)
    print("🤖 MODEL LOADER READY FOR USE")
    print("="*60)