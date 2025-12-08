import torch
from diffusers import StableDiffusionImg2ImgPipeline, EulerAncestralDiscreteScheduler
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import gc
import os
import time

class SDXLGhibliConverter:
    def __init__(self):
        print("🎨 Loading Enhanced SDXL Ghibli Converter...")
        
        # Use the most reliable model
        model_id = "runwayml/stable-diffusion-v1-5"
        
        try:
            print(f"🔄 Loading model: {model_id}")
            
            # Detect device first
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                torch.cuda.empty_cache()  # Clear cache before loading
                print("✅ Using GPU acceleration")
                dtype = torch.float16  # Use half precision for GPU to save memory
            else:
                self.device = torch.device("cpu")
                dtype = torch.float32
                print("✅ Using CPU mode")
            
            # Load model with both safetensors and bin files support
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=False,  # Set to False to allow loading .bin files
                local_files_only=False,
                variant=None,  # Don't specify variant
                ignore_mismatched_sizes=True  # Ignore size mismatches
            )
            
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations
            self.pipe.enable_attention_slicing()
            
            if self.device.type == "cuda":
                # Enable CUDA optimizations
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("✅ Enabled xformers memory efficient attention")
                except:
                    print("ℹ️  Xformers not available, continuing without")
                
                # Set CUDA device
                torch.cuda.set_device(0)
                print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
                print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            print(f"✅ Successfully loaded: {model_id}")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            # Try alternative loading method
            print("🔄 Trying alternative loading method...")
            self.load_with_alternative_method(model_id)
    
    def load_with_alternative_method(self, model_id):
        """Alternative loading method with more flexible settings"""
        try:
            print(f"🔄 Loading with alternative method: {model_id}")
            
            # Use a simpler loading approach
            from diffusers import StableDiffusionPipeline
            
            # Load basic pipeline first
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,  # Use float32 for compatibility
                safety_checker=None,
                local_files_only=False,
                revision="fp16" if torch.cuda.is_available() else "fp32"
            )
            
            # Convert to img2img pipeline
            from diffusers import StableDiffusionImg2ImgPipeline
            
            self.pipe = StableDiffusionImg2ImgPipeline(
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                tokenizer=self.pipe.tokenizer,
                unet=self.pipe.unet,
                scheduler=self.pipe.scheduler,
                safety_checker=None,
                feature_extractor=self.pipe.feature_extractor,
                requires_safety_checker=False
            )
            
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()
            
            print(f"✅ Alternative loading successful!")
            
        except Exception as e:
            print(f"❌ Alternative loading also failed: {e}")
            raise Exception(f"Could not load model: {str(e)}")

    def preprocess_image(self, image_path, target_size=768):
        """Enhanced preprocessing for better quality"""
        try:
            image = Image.open(image_path).convert("RGB")
            
            width, height = image.size
            max_dimension = max(width, height)
            
            if max_dimension > target_size:
                ratio = target_size / max_dimension
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                
                # Ensure dimensions are optimal for diffusion
                new_width = (new_width // 8) * 8
                new_height = (new_height // 8) * 8
                
                image = image.resize((new_width, new_height), Image.LANCZOS)
                print(f"📐 Resized image to {image.size} for optimal quality")
            else:
                # Optimize existing dimensions
                new_width = (width // 8) * 8
                new_height = (height // 8) * 8
                if new_width != width or new_height != height:
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                    print(f"📐 Optimized dimensions to {image.size}")
            
            return image
            
        except Exception as e:
            print(f"❌ Image preprocessing failed: {e}")
            raise

    def enhance_ghibli_style(self, image):
        """Advanced Ghibli-style enhancements"""
        try:
            print("✨ Applying advanced Ghibli enhancements...")
            
            original = image.copy()
            
            # 1. Color enhancement - Ghibli has rich, vibrant colors
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.25)
            
            # 2. Contrast enhancement for that painted look
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.15)
            
            # 3. Brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.08)
            
            # 4. Sharpness for clarity
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # 5. Soft glow effect (characteristic of Ghibli)
            blurred = image.filter(ImageFilter.GaussianBlur(1))
            image = Image.blend(image, blurred, 0.15)
            
            print("✅ Advanced enhancements applied")
            return image
            
        except Exception as e:
            print(f"⚠️ Enhancement failed: {e}")
            return image

    def create_specific_prompt(self, user_prompt, image_size):
        """Create highly specific prompts for better results"""
        print(f"🎯 Creating specific prompt for: '{user_prompt}'")
        
        # Analyze user prompt to determine the best approach
        user_prompt_lower = user_prompt.lower()
        
        # Ghibli style components
        ghibli_artistic = "Studio Ghibli style, hayao miyazaki, anime masterpiece, high quality, 4k, detailed"
        ghibli_colors = "vibrant pastel colors, soft color palette, rich saturation, beautiful lighting"
        ghibli_atmosphere = "dreamy atmosphere, whimsical, magical, enchanting, atmospheric, cinematic"
        ghibli_details = "highly detailed, intricate details, sharp focus, professional artwork"
        
        # Scene-specific enhancements
        scene_keywords = {
            'landscape': "beautiful landscape, detailed environment, scenic view, nature",
            'portrait': "character portrait, expressive face, detailed features, emotional",
            'fantasy': "fantasy world, magical creatures, enchanted, mystical",
            'city': "whimsical architecture, detailed buildings, charming cityscape",
            'animal': "cute creature, animal character, expressive eyes, friendly",
            'water': "water reflection, sparkling water, aquatic, marine"
        }
        
        # Detect scene type from user prompt
        detected_scene = "general"
        for scene_type, keywords in scene_keywords.items():
            if any(keyword in user_prompt_lower for keyword in [scene_type, *keywords.split(', ')]):
                detected_scene = scene_type
                break
        
        # Build the perfect prompt
        if user_prompt and user_prompt.strip() and user_prompt_lower not in ['', 'studio ghibli style']:
            # User provided specific prompt - enhance it
            if "ghibli" not in user_prompt_lower:
                final_prompt = (
                    f"{user_prompt}, {ghibli_artistic}, {ghibli_colors}, "
                    f"{scene_keywords.get(detected_scene, '')}, {ghibli_atmosphere}, {ghibli_details}"
                )
            else:
                final_prompt = f"{user_prompt}, {ghibli_details}, high quality, masterpiece"
        else:
            # Use default enhanced Ghibli prompt
            final_prompt = (
                f"beautiful scene, {ghibli_artistic}, {ghibli_colors}, "
                f"{ghibli_atmosphere}, {ghibli_details}, whimsical illustration"
            )
        
        # Clean up the prompt
        final_prompt = final_prompt.replace(",,", ",").strip().rstrip(',')
        
        print(f"🎨 Final enhanced prompt: {final_prompt}")
        return final_prompt

    def get_negative_prompt(self, user_prompt):
        """Get specific negative prompts based on content"""
        user_prompt_lower = user_prompt.lower()
        
        base_negative = (
            "low quality, worst quality, blurry, jpeg artifacts, ugly, "
            "bad anatomy, deformed, mutilated, disfigured, extra limbs, "
            "poorly drawn, bad proportions, cloned face, malformed, "
            "realistic, photo, 3d, watermark, signature, text, "
            "grainy, noisy, oversaturated, underexposed"
        )
        
        # Add specific negatives based on content
        if 'portrait' in user_prompt_lower or 'face' in user_prompt_lower:
            base_negative += ", poorly drawn face, asymmetric eyes, bad eyes"
        if 'landscape' in user_prompt_lower:
            base_negative += ", empty landscape, boring composition"
        if 'animal' in user_prompt_lower:
            base_negative += ", unrealistic animal, poorly drawn animal"
            
        return base_negative

    def convert_to_ghibli(self, image_path, user_prompt=""):
        """Enhanced conversion with specific style control"""
        print("🔄 Starting enhanced Ghibli conversion...")
        print(f"📁 Processing: {os.path.basename(image_path)}")
        print(f"💭 User input: '{user_prompt}'")
        
        # Clear GPU memory before starting
        self.clean_memory()
        
        try:
            # Preprocess image with better quality settings
            init_img = self.preprocess_image(image_path, 768)  # Higher resolution
            
            # Create highly specific prompt
            prompt = self.create_specific_prompt(user_prompt, init_img.size)
            negative_prompt = self.get_negative_prompt(user_prompt)

            print(f"📐 Processing at: {init_img.size}")
            print(f"🚫 Negative prompt: {negative_prompt[:100]}...")
            
            # Clear memory again
            self.clean_memory()

            # Enhanced generation parameters for better quality
            print("🎨 Generating high-quality Ghibli image...")
            start_time = time.time()
            
            # Check GPU memory before generation
            if self.device.type == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"💾 GPU Memory: {memory_used:.2f}GB / {memory_total:.2f}GB")
            
            with torch.no_grad():
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_img,
                    num_inference_steps=30,      # More steps for better quality
                    guidance_scale=8.0,          # Higher guidance for better prompt following
                    strength=0.7                 # Stronger transformation
                )

            generation_time = time.time() - start_time
            print(f"⏱️ High-quality generation completed in {generation_time:.1f}s")

            # Get result
            result_image = output.images[0]
            
            # Apply advanced enhancements
            result_image = self.enhance_ghibli_style(result_image)
            
            print("✅ Enhanced conversion completed successfully!")
            return result_image

        except torch.cuda.OutOfMemoryError:
            print("❌ GPU memory exceeded - optimizing...")
            return self.optimized_retry(image_path, user_prompt)
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            # Try optimized recovery
            try:
                print("🔄 Attempting optimized recovery...")
                return self.optimized_retry(image_path, user_prompt)
            except Exception as retry_error:
                print(f"❌ Recovery failed: {retry_error}")
                raise
        finally:
            self.clean_memory()

    def optimized_retry(self, image_path, user_prompt):
        """Optimized retry with balanced quality/speed"""
        try:
            print("🔄 Running optimized conversion...")
            
            # Use smaller but still good size
            init_img = self.preprocess_image(image_path, 512)
            
            # Slightly simplified but still specific prompt
            prompt = self.create_specific_prompt(user_prompt, init_img.size)
            negative_prompt = self.get_negative_prompt(user_prompt)
            
            print("🎨 Generating optimized Ghibli image...")
            
            # Use lower parameters to save GPU memory
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_img,
                num_inference_steps=25,
                guidance_scale=7.5,
                strength=0.65
            )
            
            result_image = output.images[0]
            result_image = self.enhance_ghibli_style(result_image)
            print("✅ Optimized conversion completed!")
            return result_image
            
        except Exception as e:
            print(f"❌ Optimized conversion failed: {e}")
            raise

    def clean_memory(self):
        """Thorough memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def save_image(self, image, filename):
        """High-quality image saving"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            if filename.lower().endswith('.png'):
                image.save(filename, optimize=True, compress_level=6)
            else:
                image.save(filename, quality=95, optimize=True, subsampling=0)
            print(f"💾 High-quality save: {filename}")
            return True
        except Exception as e:
            print(f"❌ Save failed: {e}")
            try:
                image.save(filename)
                print(f"💾 Basic save: {filename}")
                return True
            except:
                print(f"❌ All save attempts failed")
                return False