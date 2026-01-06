# monet_converter_clean.py - AVOIDS TENSORBOARD ISSUES
import os
import sys
from PIL import Image

# Prevent tensorboard from loading
sys.modules['tensorboard'] = None
sys.modules['tensorflow'] = None

class MonetConverter:
    def __init__(self, device="cpu"):
        self.device = device
        self.pipe = None
        self.load_model()
    
    def load_model(self):
        """Load model - avoids Triton conflict"""
        print("🤖 Loading AI model...")
    
        # Clear torch from cache if it exists
        if 'torch' in sys.modules:
            print("   Clearing torch cache...")
            del sys.modules['torch']
    
        # Clear triton
        if 'triton' in sys.modules:
            del sys.modules['triton']
    
        import gc
        gc.collect()
    
        try:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline
        
            print(f"   PyTorch: {torch.__version__}")
        
            # Load the model
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "Lykon/dreamshaper-8",
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
        
            if torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                print("✅ AI model loaded on CUDA!")
            else:
                self.pipe = self.pipe.to("cpu")
                self.pipe.enable_attention_slicing()
                print("✅ AI model loaded on CPU!")
            
        except Exception as e:
            print(f"⚠️ Error loading AI model: {e}")
            print("Will use simple image filters")
            self.pipe = None

    def convert(self, image_path, output_dir="monet_outputs"):
        """Convert image"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        img = Image.open(image_path).convert("RGB")
        
        # Try AI if available
        if self.pipe is not None:
            try:
                print("🎨 Using AI to create Monet style...")
                
                # Resize for AI
                img = img.resize((512, 512))
                
                prompt = (
                    "apply Claude Monet impressionist painting style to this EXACT face, "
                    "keep the same identity, same facial features, same age, same structure. "
                    "only change colors, textures and brush strokes. soft pastel colors, "
                    "impressionist brushwork, gentle lighting, canvas texture."
                )
                
                negative = (
                    "change face, distort identity, different person, deformed, blurry, "
                    "bad anatomy, extra features, unrealistic, glitch"
                )
                
                result = self.pipe(
                    prompt=prompt,
                    image=img,
                    strength=0.30,
                    guidance_scale=12,
                    negative_prompt=negative,
                    num_inference_steps=30
                ).images[0]
                
                return self.save_result(result, image_path, output_dir, "monet")
                
            except Exception as e:
                print(f"⚠️ AI failed: {e}")
                return self.simple_monet_effect(img, image_path, output_dir)
        else:
            print("🎨 Using simple Monet effect...")
            return self.simple_monet_effect(img, image_path, output_dir)
    
    def simple_monet_effect(self, img, image_path, output_dir):
        """Simple Monet effect"""
        from PIL import ImageEnhance, ImageFilter
        
        img = ImageEnhance.Color(img).enhance(1.8)
        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return self.save_result(img, image_path, output_dir, "monet")
    
    def save_result(self, img, image_path, output_dir, prefix):
        """Save image"""
        output_filename = f"{prefix}_{os.path.basename(image_path)}"
        output_path = os.path.join(output_dir, output_filename)
        img.save(output_path)
        print(f"✅ Saved: {output_path}")
        return output_path