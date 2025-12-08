import os
# Remove CPU restriction to allow GPU usage
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Comment out or remove this line

from flask import Flask, render_template, request, send_file, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from sd_xl_converter import SDXLGhibliConverter
import threading
import uuid
import secrets
import time
import psutil
import torch  # Add torch import

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

print(f"🔑 Secret Key: {app.config['SECRET_KEY']}")

# Check GPU availability
print("🔍 Checking GPU availability...")
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    print(f"🎮 GPU Detected: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.2f} GB")
    # Set CUDA device
    torch.cuda.set_device(0)
else:
    device = "cpu"
    print("⚠️  No GPU detected, using CPU")

# Global variables
conversion_status = {}
converter = None
converter_ready = False
conversion_lock = threading.Lock()

def get_system_resources():
    memory = psutil.virtual_memory()
    resources = {
        'memory_available_gb': memory.available / (1024**3),
        'memory_percent': memory.percent,
        'device': device
    }
    
    # Add GPU memory info if available
    if device == "cuda":
        try:
            gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            resources.update({
                'gpu_memory_used_gb': gpu_memory_used,
                'gpu_memory_reserved_gb': gpu_memory_reserved,
                'gpu_name': gpu_name
            })
        except:
            pass
    
    return resources

def init_converter():
    global converter, converter_ready
    print("🚀 Initializing Enhanced SDXL Ghibli Converter...")
    
    for attempt in range(3):
        try:
            resources = get_system_resources()
            print(f"💾 System resources: {resources['memory_available_gb']:.1f}GB available")
            
            # Try with smaller model first
            print("🔄 Loading converter (this may take a few minutes)...")
            converter = SDXLGhibliConverter()
            converter_ready = True
            print("✅ Enhanced SDXL Converter ready for high-quality conversions!")
            return
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                wait_time = 15 * (attempt + 1)  # Longer wait times
                print(f"🔄 Retrying in {wait_time}s...")
                time.sleep(wait_time)
                # Clear cache between attempts
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print("❌ All initialization attempts failed")
                converter = None
                converter_ready = False

# Start initialization
converter_thread = threading.Thread(target=init_converter)
converter_thread.daemon = True
converter_thread.start()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        global converter_ready, converter
        timeout = 300
        start_time = time.time()
        
        while not converter_ready and converter is None:
            if time.time() - start_time > timeout:
                return jsonify({'error': 'Model loading timeout'}), 503
            time.sleep(3)
            print(f"⏳ Waiting for enhanced model on {device}...")
        
        if converter is None:
            return jsonify({'error': 'Model failed to load'}), 503
        
        # Check active conversions
        active_conversions = len([s for s in conversion_status.values() if s['status'] == 'processing'])
        if active_conversions >= 1:  # Only one at a time for quality
            return jsonify({'error': 'Processing one image at a time for best quality'}), 503
        
        conversion_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{conversion_id}_{filename}")
        
        try:
            file.save(upload_path)
            file_size = os.path.getsize(upload_path) / (1024 * 1024)
            print(f"✅ File saved: {upload_path} ({file_size:.1f}MB)")
        except Exception as e:
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
        
        user_prompt = request.form.get('prompt', '').strip()
        print(f"📝 User prompt: '{user_prompt}'")
        
        conversion_status[conversion_id] = {
            'status': 'processing',
            'message': f'Starting high-quality Ghibli conversion on {device.upper()}...',
            'filename': filename,
            'upload_path': upload_path,
            'output_path': None,
            'prompt': user_prompt,
            'start_time': time.time(),
            'file_size_mb': file_size,
            'quality': 'enhanced',
            'device': device
        }
        
        thread = threading.Thread(target=process_conversion, args=(conversion_id, upload_path, filename, user_prompt))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'conversion_id': conversion_id,
            'message': f'File uploaded. Starting high-quality Ghibli conversion on {device.upper()}...',
            'file_size': f"{file_size:.1f}MB",
            'user_prompt': user_prompt,
            'quality': 'enhanced',
            'device': device
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

def process_conversion(conversion_id, upload_path, original_filename, user_prompt):
    try:
        with conversion_lock:
            conversion_status[conversion_id]['message'] = 'Loading image for high-quality processing...'
            
            # Check resources
            resources = get_system_resources()
            if device == "cuda":
                # Check GPU memory
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory_used > gpu_memory_total * 0.8:  # If more than 80% used
                    conversion_status[conversion_id]['message'] = 'Optimizing GPU memory for best quality...'
                    torch.cuda.empty_cache()
                    time.sleep(2)
            elif resources['memory_available_gb'] < 4:
                conversion_status[conversion_id]['message'] = 'Optimizing system memory for best quality...'
                time.sleep(3)
            
            print(f"🎨 Starting enhanced conversion for {conversion_id} on {device}")
            print(f"💡 Using prompt: '{user_prompt}'")
            
            # Clear GPU cache before conversion if using GPU
            if device == "cuda":
                torch.cuda.empty_cache()
            
            result_image = converter.convert_to_ghibli(upload_path, user_prompt)
            
            name, ext = os.path.splitext(original_filename)
            output_filename = f"ghibli_masterpiece_{name}{ext}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{conversion_id}_{output_filename}")
            
            success = converter.save_image(result_image, output_path)
            
            if success:
                processing_time = time.time() - conversion_status[conversion_id]['start_time']
                
                conversion_status[conversion_id].update({
                    'status': 'completed',
                    'message': f'High-quality Ghibli conversion completed in {processing_time:.1f}s on {device.upper()}!',
                    'output_path': output_path,
                    'output_filename': output_filename,
                    'processing_time': f"{processing_time:.1f}s",
                    'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'result_quality': 'excellent',
                    'device': device
                })
                
                print(f"✅ Enhanced conversion completed: {conversion_id} in {processing_time:.1f}s on {device}")
                
                # Clean up
                try:
                    if os.path.exists(upload_path):
                        os.remove(upload_path)
                        print(f"🧹 Cleaned up: {upload_path}")
                except:
                    pass
            else:
                conversion_status[conversion_id].update({
                    'status': 'error',
                    'message': 'Failed to save high-quality output'
                })
        
    except Exception as e:
        error_msg = f'High-quality conversion failed: {str(e)}'
        print(f"❌ {error_msg}")
        conversion_status[conversion_id].update({
            'status': 'error',
            'message': error_msg
        })
        
        try:
            if os.path.exists(upload_path):
                os.remove(upload_path)
        except:
            pass
        
    finally:
        # Clear GPU memory after conversion if using GPU
        if device == "cuda":
            torch.cuda.empty_cache()
        
        # Call converter's clean_memory method if it exists
        if hasattr(converter, 'clean_memory'):
            converter.clean_memory()

@app.route('/status/<conversion_id>')
def get_status(conversion_id):
    status = conversion_status.get(conversion_id, {
        'status': 'unknown', 
        'message': 'Conversion ID not found'
    })
    
    if status['status'] == 'processing':
        elapsed = time.time() - status.get('start_time', time.time())
        status['elapsed_time'] = f"{elapsed:.1f}s"
        status['file_size'] = f"{status.get('file_size_mb', 0):.1f}MB"
        status['quality'] = status.get('quality', 'standard')
        status['device'] = status.get('device', device)
    
    return jsonify(status)

@app.route('/download/<conversion_id>')
def download_file(conversion_id):
    status = conversion_status.get(conversion_id)
    if status and status['status'] == 'completed' and os.path.exists(status['output_path']):
        try:
            return send_file(
                status['output_path'],
                as_attachment=True,
                download_name=status['output_filename']
            )
        except Exception as e:
            return jsonify({'error': f'Download failed: {str(e)}'}), 500
    return jsonify({'error': 'File not ready'}), 404

@app.route('/preview/<conversion_id>')
def preview_file(conversion_id):
    """Preview the converted image"""
    status = conversion_status.get(conversion_id)
    if status and status['status'] == 'completed' and os.path.exists(status['output_path']):
        try:
            return send_file(
                status['output_path'],
                mimetype='image/jpeg'
            )
        except Exception as e:
            return jsonify({'error': f'Preview failed: {str(e)}'}), 500
    return jsonify({'error': 'File not ready'}), 404

@app.route('/result/<conversion_id>')
def result(conversion_id):
    status = conversion_status.get(conversion_id)
    if status:
        return render_template('result.html', 
                             conversion_id=conversion_id,
                             status=status)
    return "Conversion not found", 404

@app.route('/health')
def health_check():
    converter_status = "ready" if converter_ready else "loading"
    active = len([s for s in conversion_status.values() if s['status'] == 'processing'])
    completed = len([s for s in conversion_status.values() if s['status'] == 'completed'])
    
    resources = get_system_resources()
    
    return jsonify({
        'status': 'ok',
        'converter': converter_status,
        'conversions': {
            'active': active,
            'completed': completed,
            'total': len(conversion_status)
        },
        'system': resources,
        'quality': 'enhanced',
        'device': device,
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': gpu_name if device == "cuda" else None
    })

@app.route('/prompt-tips', methods=['GET'])
def prompt_tips():
    """Get specific prompt tips for better results"""
    tips = {
        'specific_prompts': [
            "Magical forest with glowing spirits and ancient trees - Studio Ghibli style",
            "Whimsical European village with cobblestone streets - Hayao Miyazaki art",
            "Floating islands in the sky with waterfalls and airships",
            "Japanese countryside with traditional houses and cherry blossoms",
            "Ocean view with tropical islands, clear water and marine life",
            "Steampunk city with clockwork mechanisms and airships",
            "Enchanted castle in the clouds with magical creatures",
            "Fantasy creature like Totoro in a mystical forest",
            "Young adventurer exploring ancient ruins with magical elements",
            "Witch's cottage in the woods with magical garden"
        ],
        'style_keywords': [
            "Studio Ghibli style", "Hayao Miyazaki", "anime masterpiece",
            "vibrant pastel colors", "soft cinematic lighting", "dreamy atmosphere",
            "whimsical illustration", "detailed background", "magical scenery"
        ],
        'quality_boosters': [
            "highly detailed", "masterpiece", "4k quality", "sharp focus",
            "professional artwork", "best quality", "ultra detailed"
        ]
    }
    return jsonify(tips)

@app.route('/gpu-info')
def gpu_info():
    """Get GPU information"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device': device,
    }
    
    if torch.cuda.is_available():
        info.update({
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
            'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'cuda_version': torch.version.cuda,
        })
    
    return jsonify(info)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Starting Enhanced SDXL Ghibli Converter...")
    print(f"⚡ Runtime: {device.upper()} {'🎮' if device == 'cuda' else '⚙️'}")
    if device == "cuda":
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.2f} GB")
    print("📍 Access: http://localhost:5003")
    print("🎯 Features: High-quality Ghibli transformations")
    print("💡 Tip: Use specific prompts for best results")
    print("-" * 60)
    print("📊 GPU Info: http://localhost:5003/gpu-info")
    print("🏥 Health Check: http://localhost:5003/health")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5003)