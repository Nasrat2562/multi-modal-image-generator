import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
import time
import uuid
import gc
import os
from pathlib import Path
import sys
from datetime import datetime

app = Flask(__name__, template_folder='templates', static_folder='static')

# Create organized output directories
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
IMAGE_DIR = OUTPUT_DIR / "generated_images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Model cache path
MODEL_CACHE_PATH = Path.home() / ".cache" / "huggingface" / "hub" / "models--runwayml--stable-diffusion-v1-5"

print("=" * 60)
print("🚀 AI Image Generator Starting...")
print("=" * 60)
print(f"📁 Output folder: {OUTPUT_DIR}")
print(f"🖼️  Images will be saved to: {IMAGE_DIR}")

class UltimatePromptToImage:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"💻 Device: {self.device.upper()}")
        print(f"🎯 Precision: {'FP16' if self.dtype == torch.float16 else 'FP32'}")
        
        # Check if model is cached
        if MODEL_CACHE_PATH.exists():
            print("✅ Model found in cache")
            self.model_downloaded = True
        else:
            print("⚠️ Model not cached - will download on first use")
            self.model_downloaded = False
        
        self.pipe = None
        self.model_loaded = False
        
        # Generation presets
        self.presets = {
            "fast": {"steps": 20, "guidance": 7.5, "size": (512, 512)},
            "standard": {"steps": 25, "guidance": 8.0, "size": (512, 512)},
            "quality": {"steps": 30, "guidance": 8.5, "size": (512, 512)}
        }
    
    def load_model(self):
        """Load the Stable Diffusion model"""
        if self.model_loaded:
            return True
            
        try:
            if not self.model_downloaded:
                print("📥 Downloading AI model for the first time...")
                print("⚠️ This may take 10-30 minutes (5GB)")
                print("💾 Will be cached for future use")
            else:
                print("📥 Loading AI model from cache...")
            
            start_time = time.time()
            
            # Load scheduler
            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                subfolder="scheduler",
                solver_order=2,
                algorithm_type="dpmsolver++"
            )
            
            # Load pipeline
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=self.dtype,
                scheduler=scheduler,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Apply optimizations
            if self.device == "cuda":
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("✅ XFormers enabled")
                except:
                    print("⚠️ XFormers not available")
            
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()
            
            self.model_loaded = True
            load_time = time.time() - start_time
            
            if not self.model_downloaded:
                print(f"✅ Model downloaded and loaded in {load_time:.1f} seconds")
                print("🎉 Future runs will be much faster!")
                self.model_downloaded = True
            else:
                print(f"✅ Model loaded from cache in {load_time:.1f} seconds")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def enhance_prompt(self, user_prompt, style="realistic"):
        """Enhance user prompts"""
        style_enhancements = {
            "realistic": "photorealistic, highly detailed, professional photography, sharp focus, 8k",
            "anime": "anime artwork, Japanese animation, vibrant colors, cel shading, detailed eyes",
            "fantasy": "fantasy art, epic, magical, mystical, dreamlike, concept art",
            "digital_art": "digital painting, concept art, illustration, trending on artstation",
            "cinematic": "cinematic, movie still, dramatic lighting, film grain, wide angle",
            "painting": "oil painting, brush strokes, canvas texture, artistic",
            "scifi": "sci-fi, futuristic, cyberpunk, neon, advanced technology",
            "minimalist": "minimalist, simple, clean, elegant, modern"
        }
        
        quality_boost = "masterpiece, best quality, ultra detailed, high resolution"
        style_terms = style_enhancements.get(style, "high quality, detailed")
        
        return f"{user_prompt}, {style_terms}, {quality_boost}"
    
    def get_negative_prompt(self, style="realistic"):
        """Get negative prompt"""
        base_negative = (
            "poor quality, low quality, blurry, pixelated, distorted, ugly, "
            "bad anatomy, bad proportions, extra limbs, disfigured, deformed, "
            "mutation, mutated, watermark, signature, text, error"
        )
        
        style_negatives = {
            "realistic": "3d render, cartoon, anime, painting, drawing, artificial",
            "anime": "photorealistic, realistic, photo, live action, 3d render",
            "fantasy": "realistic, photograph, mundane, ordinary, boring"
        }
        
        additional = style_negatives.get(style, "")
        return f"{base_negative}, {additional}" if additional else base_negative
    
    def generate_image(self, prompt, style="realistic", preset="standard"):
        """Generate a single image"""
        if not self.model_loaded:
            if not self.load_model():
                return None, "Model failed to load", 0
        
        # Get preset parameters
        params = self.presets.get(preset, self.presets["standard"])
        
        # Enhance prompts
        enhanced_prompt = self.enhance_prompt(prompt, style)
        negative_prompt = self.get_negative_prompt(style)
        
        print(f"🎨 Generating: {prompt[:50]}...")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate image
        try:
            start_time = time.time()
            
            with torch.no_grad():
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=params["steps"],
                    guidance_scale=params["guidance"],
                    width=params["size"][0],
                    height=params["size"][1],
                    num_images_per_prompt=1
                )
            
            generation_time = time.time() - start_time
            image = result.images[0]
            
            print(f"✅ Generated in {generation_time:.1f}s")
            return image, enhanced_prompt, generation_time
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return None, str(e), 0

# Initialize generator
generator = UltimatePromptToImage()

# Flask Routes
@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Check API status"""
    return jsonify({
        "ready": generator.model_loaded,
        "device": generator.device,
        "model_loaded": generator.model_loaded,
        "status": "ready" if generator.model_loaded else "loading_model",
        "output_folder": str(OUTPUT_DIR)
    })

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Generate image endpoint"""
    try:
        data = request.get_json()
        
        prompt = data.get('prompt', '').strip()
        style = data.get('style', 'realistic')
        preset = data.get('preset', 'standard')
        
        if not prompt:
            return jsonify({"success": False, "error": "Please enter a prompt"}), 400
        
        print(f"📨 Received: {prompt[:50]}...")
        
        # Generate image
        image, enhanced_prompt, gen_time = generator.generate_image(
            prompt=prompt,
            style=style,
            preset=preset
        )
        
        if image is None:
            return jsonify({"success": False, "error": enhanced_prompt}), 500
        
        # Create descriptive filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"ai_image_{timestamp}_{safe_prompt}_{style}.png"
        filepath = IMAGE_DIR / filename
        
        # Save image with metadata
        image.save(filepath, quality=95, optimize=True)
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return jsonify({
            "success": True,
            "images": [f"/api/image/{filename}"],
            "download_url": f"/api/download/{filename}",
            "filename": filename,
            "filepath": str(filepath),
            "enhanced_prompt": enhanced_prompt,
            "time": round(gen_time, 2),
            "prompt": prompt,
            "style": style,
            "preset": preset,
            "device": generator.device
        })
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/image/<filename>')
def serve_image(filename):
    """Serve generated images for display"""
    try:
        return send_from_directory(IMAGE_DIR, filename)
    except:
        return jsonify({"error": "Image not found"}), 404

@app.route('/api/download/<filename>')
def download_image(filename):
    """Download generated image"""
    try:
        filepath = IMAGE_DIR / filename
        if not filepath.exists():
            return jsonify({"error": "File not found"}), 404
        
        # Create a friendly download filename
        download_name = filename.replace("ai_image_", "ai_art_")
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=download_name,
            mimetype='image/png'
        )
    except Exception as e:
        print(f"❌ Download error: {e}")
        return jsonify({"error": "Download failed"}), 500

@app.route('/api/styles')
def api_styles():
    """Get available styles"""
    styles = ["realistic", "anime", "fantasy", "digital_art", 
              "cinematic", "painting", "scifi", "minimalist"]
    return jsonify({"styles": styles})

@app.route('/api/presets')
def api_presets():
    """Get available presets"""
    presets = list(generator.presets.keys())
    return jsonify({"presets": presets})

@app.route('/api/list_images')
def list_images():
    """List all generated images"""
    try:
        images = []
        for file in IMAGE_DIR.glob("*.png"):
            images.append({
                "filename": file.name,
                "url": f"/api/image/{file.name}",
                "download_url": f"/api/download/{file.name}",
                "size": file.stat().st_size,
                "created": datetime.fromtimestamp(file.stat().st_ctime).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Sort by creation time (newest first)
        images.sort(key=lambda x: x["created"], reverse=True)
        
        return jsonify({
            "success": True,
            "count": len(images),
            "images": images,
            "folder": str(IMAGE_DIR)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/health')
def health_check():
    """Health check"""
    return jsonify({
        "status": "running",
        "model_loaded": generator.model_loaded,
        "device": generator.device,
        "port": 5001,
        "output_folder": str(OUTPUT_DIR),
        "images_count": len(list(IMAGE_DIR.glob("*.png")))
    })

if __name__ == '__main__':
    print(f"🌐 Server running on port 5001")
    print(f"📁 Output folder: {OUTPUT_DIR}")
    print(f"🖼️  Images folder: {IMAGE_DIR}")
    
    # Load model in background
    import threading
    def load_model_background():
        print("🔧 Loading AI model in background...")
        generator.load_model()
    
    thread = threading.Thread(target=load_model_background)
    thread.daemon = True
    thread.start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True, use_reloader=False)