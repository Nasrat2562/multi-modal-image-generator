# toon_converter.py - SIMPLIFIED VERSION (AVOIDS TRITON CONFLICT)
import os
import sys
import time
import traceback
from PIL import Image, ImageEnhance, ImageFilter
import uuid

print("\n" + "="*60)
print("✨ TOON CONVERTER INITIALIZED (SIMPLE MODE)")
print("="*60)

class ToonConverter:
    def __init__(self):
        self.pipe = None
        self.model_loaded = False
        self.device = "cpu"
        
        # Try to load AI model once
        self._try_load_ai_model()
    
    def _try_load_ai_model(self):
        """Try to load AI model without causing Triton conflicts"""
        print("\n🤖 ATTEMPTING TO LOAD AI MODEL...")
        
        # Add a delay to avoid rapid re-imports
        time.sleep(1)
        
        try:
            # IMPORTANT: Clear torch from sys.modules if it exists
            if 'torch' in sys.modules:
                print("   ⚠️ Clearing torch from cache...")
                del sys.modules['torch']
            
            # Clear other related modules
            modules_to_clear = ['triton', 'diffusers', 'transformers']
            for module in modules_to_clear:
                if module in sys.modules:
                    try:
                        del sys.modules[module]
                    except:
                        pass
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Now import torch fresh
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline
            
            print(f"   ✅ PyTorch {torch.__version__} loaded")
            
            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"   Device: {self.device}")
            
            # Load model (skip if already tried recently)
            print("   Loading Lykon/dreamshaper-8...")
            
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "Lykon/dreamshaper-8",
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.pipe = self.pipe.to(self.device)
            
            if self.device == "cpu":
                self.pipe.enable_attention_slicing()
            
            self.model_loaded = True
            print("   ✅ AI Model loaded successfully!")
            
        except Exception as e:
            print(f"   ❌ AI Model failed to load: {e}")
            print("   Will use simple cartoon filters")
            self.pipe = None
            self.model_loaded = False
    
    def convert(self, image_path, output_dir):
        """Convert image with fallback to simple filters"""
        print(f"\n🖼 Processing: {os.path.basename(image_path)}")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Open and prepare image
            img = Image.open(image_path).convert("RGB")
            img_resized = img.resize((512, 512))
            
            # Try AI conversion if available
            if self.model_loaded and self.pipe is not None:
                print("🎨 Attempting AI conversion...")
                
                try:
                    prompt = (
                        "high quality toon-style portrait of the same person, smooth skin, "
                        "clean lines, Pixar + anime blend, vibrant colors, cute aesthetic"
                    )
                    
                    negative = "realistic skin, wrinkles, bad proportions, blur, distorted face"
                    
                    result = self.pipe(
                        prompt=prompt,
                        image=img_resized,
                        strength=0.6,
                        guidance_scale=8,
                        negative_prompt=negative,
                        num_inference_steps=30
                    ).images[0]
                    
                    output_filename = f"toon_ai_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
                    output_path = os.path.join(output_dir, output_filename)
                    result.save(output_path)
                    
                    print(f"✅ AI conversion successful!")
                    return output_path
                    
                except Exception as ai_error:
                    print(f"⚠️ AI conversion failed: {ai_error}")
                    print("   Falling back to simple filter...")
            
            # Simple cartoon filter (always works)
            print("🎨 Using enhanced cartoon filter...")
            
            # Multiple cartoon effects
            img = img.resize((512, 512))
            
            # 1. Enhance colors
            img = ImageEnhance.Color(img).enhance(2.2)
            
            # 2. Increase contrast
            img = ImageEnhance.Contrast(img).enhance(1.6)
            
            # 3. Smooth the image (cartoon effect)
            img = img.filter(ImageFilter.SMOOTH_MORE)
            img = img.filter(ImageFilter.SMOOTH_MORE)
            
            # 4. Sharpen edges
            img = img.filter(ImageFilter.SHARPEN)
            
            # 5. Create outline effect
            edges = img.filter(ImageFilter.FIND_EDGES).convert("L")
            edges = ImageEnhance.Brightness(edges).enhance(3.0)
            
            # Convert back to RGB and combine with edges
            edges_rgb = edges.convert("RGB")
            
            # Create final image with black outlines
            from PIL import ImageChops
            
            # Darken the edges
            edges_dark = ImageEnhance.Brightness(edges_rgb).enhance(0.3)
            
            # Combine with original
            result_img = ImageChops.multiply(img, edges_dark)
            
            # Final enhancement
            result_img = ImageEnhance.Color(result_img).enhance(1.2)
            
            output_filename = f"toon_filter_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
            output_path = os.path.join(output_dir, output_filename)
            result_img.save(output_path)
            
            print(f"✅ Cartoon filter applied!")
            return output_path
            
        except Exception as e:
            print(f"❌ Error in conversion: {e}")
            traceback.print_exc()
            
            # Last resort: just save the original
            output_filename = f"toon_original_{os.path.basename(image_path)}"
            output_path = os.path.join(output_dir, output_filename)
            Image.open(image_path).save(output_path)
            return output_path

# For testing
if __name__ == "__main__":
    converter = ToonConverter()
    print(f"\n✅ ToonConverter initialized")
    print(f"   AI Model loaded: {converter.model_loaded}")
    print(f"   Device: {converter.device}")