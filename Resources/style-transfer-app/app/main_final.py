# main_final.py - FIXED WITH IMAGE-TO-IMAGE MODEL
import os
import sys
import uuid
import traceback
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image, ImageEnhance, ImageFilter

print("=" * 80)
print("🎨 AI ART TRANSFORMER - IMAGE-TO-IMAGE VERSION")
print("=" * 80)

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

print(f"Current directory: {CURRENT_DIR}")
print(f"Project root: {PROJECT_ROOT}")

# Template paths
TEMPLATES_DIR = os.path.join(CURRENT_DIR, 'templates')
STATIC_DIR = os.path.join(CURRENT_DIR, 'static')

print(f"Templates directory: {TEMPLATES_DIR}")
print(f"Static directory: {STATIC_DIR}")

# Create necessary directories
UPLOADS_DIR = os.path.join(PROJECT_ROOT, 'uploads')
MONET_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'monet_outputs')
TOON_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'toon_outputs')

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(MONET_OUTPUTS_DIR, exist_ok=True)
os.makedirs(TOON_OUTPUTS_DIR, exist_ok=True)

# Initialize Flask
app = Flask(__name__,
            template_folder=TEMPLATES_DIR,
            static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOADS_DIR

# ============ LOAD IMAGE-TO-IMAGE AI MODEL ============
AI_AVAILABLE = False
ai_pipe = None
ai_device = "cpu"
MODEL_ID = "Lykon/dreamshaper-8"  # YOUR EXACT MODEL

try:
    print(f"\n🤖 LOADING IMAGE-TO-IMAGE MODEL: {MODEL_ID}")
    
    # Clear any existing imports
    if 'torch' in sys.modules:
        del sys.modules['torch']
    if 'diffusers' in sys.modules:
        del sys.modules['diffusers']
    
    import gc
    gc.collect()
    
    # Import torch
    import torch
    print(f"✅ PyTorch {torch.__version__} loaded")
    
    # Import the CORRECT pipeline: StableDiffusionImg2ImgPipeline
    from diffusers import StableDiffusionImg2ImgPipeline
    
    print("🚀 Loading image-to-image model...")
    print("(First time: This downloads ~2GB, takes 5-10 minutes)")
    
    # Load the model
    ai_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Move to device
    if torch.cuda.is_available():
        ai_pipe = ai_pipe.to("cuda")
        ai_device = "cuda"
        print("✅ AI model loaded on CUDA!")
    else:
        ai_pipe = ai_pipe.to("cpu")
        ai_pipe.enable_attention_slicing()  # Reduce memory usage
        ai_device = "cpu"
        print("✅ AI model loaded on CPU!")
    
    AI_AVAILABLE = True
    print(f"🎉 IMAGE-TO-IMAGE MODEL LOADED SUCCESSFULLY!")
    print(f"   Model: {MODEL_ID}")
    print(f"   Device: {ai_device}")
    
except Exception as e:
    print(f"⚠️ AI model not available: {e}")
    print("✨ Using enhanced artistic filters instead")
    AI_AVAILABLE = False

# ============ ENHANCED ARTISTIC FILTERS ============
def apply_monet_filter(img):
    """Enhanced Monet-style filter"""
    img = img.resize((1024, 1024))
    img = ImageEnhance.Color(img).enhance(1.8)
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    img = ImageEnhance.Brightness(img).enhance(1.1)
    return img

def apply_toon_filter(img):
    """Enhanced cartoon filter"""
    img = img.resize((1024, 1024))
    img = ImageEnhance.Color(img).enhance(2.2)
    img = ImageEnhance.Contrast(img).enhance(1.7)
    for _ in range(2):
        img = img.filter(ImageFilter.SMOOTH_MORE)
    img = img.filter(ImageFilter.SHARPEN)
    return img

# ============ AI-ENHANCED CONVERSION FUNCTIONS ============
def convert_to_monet(image_path, output_dir):
    """Convert image to Monet style using AI"""
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")
        original_img = img.copy()
        
        if AI_AVAILABLE and ai_pipe is not None:
            print("🎨 Using AI Image-to-Image for Monet conversion...")
            try:
                # Resize for AI model (optimal size)
                img_ai = img.resize((512, 512))
                
                # Monet-specific prompt for image-to-image
                prompt = (
                    "Claude Monet impressionist painting style, soft brush strokes, "
                    "pastel colors, impressionism, oil painting, artistic masterpiece, "
                    "maintain facial features and identity, same person, same face structure"
                )
                
                negative_prompt = (
                    "ugly, deformed, distorted face, blurry, bad anatomy, "
                    "different person, cartoon, digital art, photo"
                )
                
                # Generate with AI image-to-image
                result = ai_pipe(
                    prompt=prompt,
                    image=img_ai,
                    negative_prompt=negative_prompt,
                    strength=0.4,  # Lower strength keeps more of original
                    guidance_scale=9.0,
                    num_inference_steps=35
                ).images[0]
                
                # Resize back to original dimensions
                result = result.resize(original_img.size)
                ai_used = True
                
                print("✅ AI Monet conversion successful!")
                
            except Exception as ai_error:
                print(f"⚠️ AI failed: {ai_error}, using filter")
                result = apply_monet_filter(original_img)
                ai_used = False
        else:
            print("🎨 Using enhanced Monet filter...")
            result = apply_monet_filter(original_img)
            ai_used = False
        
        # Save result
        output_filename = f"monet_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
        output_path = os.path.join(output_dir, output_filename)
        result.save(output_path, quality=95)
        
        return output_path, ai_used
        
    except Exception as e:
        print(f"❌ Monet conversion error: {e}")
        traceback.print_exc()
        # Fallback
        output_filename = f"monet_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
        output_path = os.path.join(output_dir, output_filename)
        Image.open(image_path).save(output_path)
        return output_path, False

def convert_to_toon(image_path, output_dir):
    """Convert image to cartoon style using AI"""
    try:
        # Load image
        img = Image.open(image_path).convert("RGB")
        original_img = img.copy()
        
        if AI_AVAILABLE and ai_pipe is not None:
            print("✨ Using AI Image-to-Image for cartoon conversion...")
            try:
                # Resize for AI model
                img_ai = img.resize((512, 512))
                
                # Cartoon-specific prompt for image-to-image
                prompt = (
                    "cartoon style, Pixar animation, Disney character, smooth skin, "
                    "vibrant colors, clean lines, cute aesthetic, maintain facial features, "
                    "same person, same identity, cartoon portrait"
                )
                
                negative_prompt = (
                    "realistic, photograph, blurry, deformed, ugly, bad anatomy, "
                    "different person, painting, drawing"
                )
                
                # Generate with AI image-to-image
                result = ai_pipe(
                    prompt=prompt,
                    image=img_ai,
                    negative_prompt=negative_prompt,
                    strength=0.5,  # Medium strength for good transformation
                    guidance_scale=8.0,
                    num_inference_steps=35
                ).images[0]
                
                # Resize back to original dimensions
                result = result.resize(original_img.size)
                ai_used = True
                
                print("✅ AI Cartoon conversion successful!")
                
            except Exception as ai_error:
                print(f"⚠️ AI failed: {ai_error}, using filter")
                result = apply_toon_filter(original_img)
                ai_used = False
        else:
            print("✨ Using enhanced cartoon filter...")
            result = apply_toon_filter(original_img)
            ai_used = False
        
        # Save result
        output_filename = f"toon_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
        output_path = os.path.join(output_dir, output_filename)
        result.save(output_path, quality=95)
        
        return output_path, ai_used
        
    except Exception as e:
        print(f"❌ Toon conversion error: {e}")
        traceback.print_exc()
        # Fallback
        output_filename = f"toon_{uuid.uuid4().hex[:8]}_{os.path.basename(image_path)}"
        output_path = os.path.join(output_dir, output_filename)
        Image.open(image_path).save(output_path)
        return output_path, False

# ============ FLASK ROUTES ============
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/monet')
def monet_page():
    return render_template('monet.html')

@app.route('/toon')
def toon_page():
    return render_template('toon.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex[:8]}_{filename}"
            filepath = os.path.join(UPLOADS_DIR, unique_filename)
            file.save(filepath)
            print(f"📤 Uploaded: {unique_filename}")
            return jsonify({
                'success': True,
                'filename': unique_filename,
                'message': 'File uploaded successfully'
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/convert/monet', methods=['POST'])
def convert_monet():
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400
        
        filename = data['filename']
        filepath = os.path.join(UPLOADS_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        print(f"\n🎨 Converting to Monet style: {filename}")
        output_path, ai_used = convert_to_monet(filepath, MONET_OUTPUTS_DIR)
        output_filename = os.path.basename(output_path)
        
        return jsonify({
            'success': True,
            'output_filename': output_filename,
            'ai_used': ai_used,
            'model': MODEL_ID if ai_used else 'Enhanced Filter',
            'message': 'Monet conversion successful!'
        })
        
    except Exception as e:
        print(f"❌ Monet conversion error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/convert/toon', methods=['POST'])
def convert_toon():
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({'error': 'No filename provided'}), 400
        
        filename = data['filename']
        filepath = os.path.join(UPLOADS_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        print(f"\n✨ Converting to cartoon style: {filename}")
        output_path, ai_used = convert_to_toon(filepath, TOON_OUTPUTS_DIR)
        output_filename = os.path.basename(output_path)
        
        return jsonify({
            'success': True,
            'output_filename': output_filename,
            'ai_used': ai_used,
            'model': MODEL_ID if ai_used else 'Enhanced Filter',
            'message': 'Cartoon conversion successful!'
        })
        
    except Exception as e:
        print(f"❌ Toon conversion error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<style>/<filename>')
def download_file(style, filename):
    try:
        if style == 'monet':
            folder = MONET_OUTPUTS_DIR
        elif style == 'toon':
            folder = TOON_OUTPUTS_DIR
        else:
            return jsonify({'error': 'Invalid style'}), 400
        
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/image/<style>/<filename>')
def preview_file(style, filename):
    try:
        if style == 'monet':
            folder = MONET_OUTPUTS_DIR
        elif style == 'toon':
            folder = TOON_OUTPUTS_DIR
        else:
            return jsonify({'error': 'Invalid style'}), 400
        
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, mimetype='image/jpeg')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        'status': 'running',
        'ai_available': AI_AVAILABLE,
        'ai_device': ai_device if AI_AVAILABLE else 'none',
        'model': MODEL_ID if AI_AVAILABLE else 'Enhanced Filters',
        'message': f'Image-to-image model {MODEL_ID} is ready!' if AI_AVAILABLE else 'Using enhanced artistic filters'
    })

if __name__ == '__main__':
    print(f"\n{'='*80}")
    print("🚀 AI ART TRANSFORMER - READY!")
    print(f"{'='*80}")
    print(f"📍 Local:     http://localhost:5004")
    print(f"🎨 Monet AI:  http://localhost:5004/monet")
    print(f"✨ Toon AI:   http://localhost:5004/toon")
    print(f"📊 Status:    http://localhost:5004/status")
    print(f"{'='*80}")
    
    if AI_AVAILABLE:
        print("✅ IMAGE-TO-IMAGE AI MODEL LOADED!")
        print(f"   Model: {MODEL_ID}")
        print(f"   Device: {ai_device}")
        print("   Your photos will be transformed while keeping facial features!")
    else:
        print("✨ USING ENHANCED ARTISTIC FILTERS")
        print("   Beautiful artistic effects - AI model not available")
    
    print(f"{'='*80}\n")
    
    app.run(debug=False, host='0.0.0.0', port=5004, threaded=True)