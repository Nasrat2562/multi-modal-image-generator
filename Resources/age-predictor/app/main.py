import os
import uuid
import json
import hashlib
import secrets
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
from age_detector import AgeDetector

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'age-predictor-secret'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Set absolute paths
UPLOAD_FOLDER = os.path.join(BASE_DIR, app.config['UPLOAD_FOLDER'])
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
USERS_DB_FILE = os.path.join(BASE_DIR, 'users.json')

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Initialize detector
age_detector = AgeDetector()

# Debug print
print(f"📁 Current directory: {BASE_DIR}")
print(f"📁 Template folder: {app.template_folder}")

# ==================== LOCAL AUTHENTICATION FUNCTIONS ====================

def init_users_db():
    """Initialize user database file if it doesn't exist"""
    if not os.path.exists(USERS_DB_FILE):
        print(f"📝 Creating new users database at: {USERS_DB_FILE}")
        with open(USERS_DB_FILE, 'w') as f:
            json.dump([], f)
    else:
        # Check if database is valid JSON
        try:
            with open(USERS_DB_FILE, 'r') as f:
                data = json.load(f)
                print(f"✅ Users database loaded, {len(data)} users found")
        except json.JSONDecodeError:
            print(f"❌ Users database corrupted, recreating...")
            with open(USERS_DB_FILE, 'w') as f:
                json.dump([], f)

def load_users():
    """Load users from JSON file"""
    try:
        with open(USERS_DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_users(users):
    """Save users to JSON file"""
    with open(USERS_DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    """Hash password using SHA-256 with salt"""
    salt = secrets.token_hex(8)
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}${hashed}"

def verify_password(password, hashed_password):
    """Verify password against hash"""
    if not hashed_password:
        return False
    try:
        salt, stored_hash = hashed_password.split('$')
        computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return computed_hash == stored_hash
    except:
        return False

# Initialize user database on startup
init_users_db()

# ==================== HELPER FUNCTIONS ====================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# ==================== ROUTES ====================

@app.route('/')
def login():
    """Login page"""
    return render_template('login.html')

@app.route('/face-verification')
def face_verification():
    """Face verification page"""
    return render_template('face_verification.html')

@app.route('/age-predictor')
def age_predictor_page():
    """Standalone age predictor page"""
    return render_template('index.html')

@app.route('/signup')
def signup_page():
    """Render signup page"""
    print(f"🎯 Rendering signup.html")
    return render_template('signup.html')

@app.route('/verify-face', methods=['POST'])
def verify_face():
    """Face verification endpoint"""
    print("=== /verify-face endpoint called ===")
    
    try:
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({'success': False, 'error': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'})
        
        print(f"✅ File received: {file.filename}")
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = f"face_verify_{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            print(f"✅ File saved to: {filepath}")
            
            if not os.path.exists(filepath):
                return jsonify({'success': False, 'error': 'Failed to save image'})
            
            print("🔍 Starting age verification...")
            result = age_detector.verify_age(filepath, RESULTS_FOLDER)
            print(f"✅ Age verification result: {result}")
            
            if result['success']:
                # Store verification data in session
                session['face_verified'] = True
                session['user_age'] = result.get('age', 0)
                session['verification_status'] = result.get('verification_status', 'UNKNOWN')
                session['faces_detected'] = result.get('faces_detected', 0)
                
                response_data = {
                    'success': True,
                    'age': result.get('age', 0),
                    'verified': result.get('verified', False),
                    'verification_status': result.get('verification_status', 'UNKNOWN'),
                    'verification_message': result.get('verification_message', ''),
                    'faces_detected': result.get('faces_detected', 0),
                    'confidence': result.get('confidence', 'N/A'),
                    'redirect': '/dashboard'
                }
                print(f"📊 Sending response: {response_data}")
                return jsonify(response_data)
            else:
                error_msg = result.get('error', 'Unknown error during verification')
                print(f"❌ Verification failed: {error_msg}")
                return jsonify({'success': False, 'error': error_msg})
        
        else:
            print("❌ Invalid file type")
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload JPG, PNG, or GIF.'})
            
    except Exception as e:
        print(f"💥 Unexpected error in verify_face: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': unique_name,
            'message': 'File uploaded successfully'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict', methods=['POST'])
def predict_age():
    data = request.json
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    result = age_detector.predict_age(filepath, RESULTS_FOLDER)
    
    if result['success']:
        return jsonify({
            'success': True,
            'age': result['age'],
            'faces_detected': result['faces_detected'],
            'output_image': result['output_image'],
            'confidence': result.get('confidence', 'N/A'),
            'message': 'Age prediction successful'
        })
    else:
        return jsonify({'error': result['error']}), 500

@app.route('/results/<filename>')
def get_result(filename):
    """Serve result images"""
    try:
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error serving image: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_result(filename):
    """Download result images"""
    try:
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(
                file_path, 
                as_attachment=True,
                download_name=filename,
                mimetype='image/jpeg'
            )
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'}), 500

@app.route('/dashboard')
def dashboard():
    """Dashboard page after successful verification"""
    print(f"📊 Dashboard access attempt")
    print(f"📊 Session: face_verified={session.get('face_verified')}, logged_in={session.get('logged_in')}")
    
    # Check if user is authenticated
    if session.get('face_verified', False) or session.get('logged_in', False):
        print(f"✅ User authenticated, rendering dashboard")
        return render_template('dashboard.html')
    else:
        print(f"❌ User not authenticated, checking localStorage fallback...")
        # Check localStorage via query parameter or just redirect
        return redirect('/')

@app.route('/dashboard.html')
def dashboard_redirect():
    """Redirect dashboard.html requests to /dashboard"""
    return redirect('/dashboard')

# ==================== NEW SIGNUP/LOGIN API ROUTES ====================

@app.route('/api/signup', methods=['POST'])
def api_signup():
    """Handle user signup with local authentication"""
    try:
        data = request.json
        full_name = data.get('fullName', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        age = data.get('age')
        
        print(f"🔐 Signup attempt for: {email}")
        
        # Validation
        if not all([full_name, email, password, age]):
            return jsonify({'success': False, 'error': 'All fields are required'})
        
        if len(password) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters'})
        
        if int(age) < 18:
            return jsonify({'success': False, 'error': 'You must be 18 or older to sign up'})
        
        # Check if email already exists
        users = load_users()
        for user in users:
            if user['email'] == email:
                return jsonify({'success': False, 'error': 'Email already registered'})
        
        # Create new user
        new_user = {
            'id': secrets.token_hex(16),
            'full_name': full_name,
            'email': email,
            'password_hash': hash_password(password),
            'age': int(age),
            'face_verified': True,
            'created_at': datetime.utcnow().isoformat(),
            'last_login': datetime.utcnow().isoformat(),
            'status': 'active'
        }
        
        users.append(new_user)
        save_users(users)
        
        print(f"✅ User created: {email}, ID: {new_user['id']}")
        
        # IMPORTANT: Set up Flask session for immediate login
        session['user_id'] = new_user['id']
        session['user_email'] = new_user['email']
        session['user_name'] = new_user['full_name']
        session['user_age'] = new_user['age']
        session['logged_in'] = True
        session['face_verified'] = True
        
        print(f"✅ Session created for user: {session['user_email']}")
        
        return jsonify({
            'success': True,
            'message': 'Account created successfully',
            'user': {
                'id': new_user['id'],
                'name': full_name,
                'email': email,
                'age': age
            },
            'redirect': '/dashboard'
        })
        
    except Exception as e:
        print(f"❌ Signup error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400
@app.route('/api/login', methods=['POST'])
def api_login():
    """Handle user login with local authentication"""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        print(f"🔐 Login attempt for email: {email}")
        
        if not email or not password:
            print(f"❌ Missing email or password")
            return jsonify({'success': False, 'error': 'Email and password required'})
        
        # Find user by email
        users = load_users()
        print(f"📋 Total users in database: {len(users)}")
        
        user = None
        for u in users:
            print(f"  Checking user: {u['email']}")
            if u['email'] == email:
                user = u
                break
        
        if not user:
            print(f"❌ User not found: {email}")
            return jsonify({'success': False, 'error': 'Invalid email or password'})
        
        print(f"✅ User found: {user['email']}")
        
        # Verify password
        if not verify_password(password, user.get('password_hash')):
            print(f"❌ Password verification failed for: {email}")
            return jsonify({'success': False, 'error': 'Invalid email or password'})
        
        print(f"✅ Password verified successfully for: {email}")
        
        # Update last login
        user['last_login'] = datetime.utcnow().isoformat()
        save_users(users)
        
        # Create session
        session['user_id'] = user['id']
        session['user_email'] = user['email']
        session['user_name'] = user['full_name']
        session['user_age'] = user['age']
        session['logged_in'] = True
        session['face_verified'] = True
        
        print(f"✅ Session created for user: {session['user_email']}")
        print(f"✅ Session data: user_id={session['user_id']}, logged_in={session['logged_in']}")
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'user': {
                'id': user['id'],
                'name': user['full_name'],
                'email': user['email'],
                'age': user['age']
            }
        })
        
    except Exception as e:
        print(f"❌ Login error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 401

@app.route('/api/check-email', methods=['POST'])
def check_email():
    """Check if email already exists"""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        
        if not email:
            return jsonify({'exists': False, 'valid': False})
        
        users = load_users()
        exists = any(user['email'] == email for user in users)
        
        return jsonify({
            'exists': exists,
            'message': 'Email already registered' if exists else 'Email available'
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Logout user"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/user', methods=['GET'])
def get_current_user():
    """Get current user data"""
    if 'user_id' not in session:
        return jsonify({'authenticated': False})
    
    return jsonify({
        'authenticated': True,
        'user': {
            'id': session.get('user_id'),
            'name': session.get('user_name'),
            'email': session.get('user_email'),
            'age': session.get('user_age')
        }
    })

@app.route('/check-auth')
def check_auth():
    """Check if user is authenticated"""
    print(f"🔍 Checking authentication...")
    
    # Check traditional login (face verification)
    if session.get('face_verified', False):
        print(f"✅ User authenticated via face verification")
        return jsonify({
            'authenticated': True,
            'method': 'face_verification',
            'age': session.get('user_age', 0)
        })
    
    # Check signup/login authentication
    elif session.get('logged_in', False):
        print(f"✅ User authenticated via signup/login")
        return jsonify({
            'authenticated': True,
            'method': 'signup',
            'age': session.get('user_age', 0),
            'user': {
                'name': session.get('user_name'),
                'email': session.get('user_email')
            }
        })
    
    else:
        print(f"❌ User not authenticated")
        return jsonify({
            'authenticated': False,
            'method': 'none'
        })

@app.route('/logout')
def logout():
    """Clear session and logout"""
    session.clear()
    return redirect('/')

# ==================== DEBUG ROUTE ====================

@app.route('/debug-paths')
def debug_paths():
    """Debug route to check file paths"""
    template_dir = os.path.join(BASE_DIR, 'templates')
    files = os.listdir(template_dir) if os.path.exists(template_dir) else []
    
    return jsonify({
        'base_dir': BASE_DIR,
        'template_folder': app.template_folder,
        'full_template_path': template_dir,
        'templates_exist': os.path.exists(template_dir),
        'files_in_templates': files,
        'signup_exists': 'signup.html' in files,
        'users_db_path': USERS_DB_FILE,
        'users_db_exists': os.path.exists(USERS_DB_FILE)
    })

@app.route('/debug-users')
def debug_users():
    """Debug route to check all users"""
    users = load_users()
    return jsonify({
        'total_users': len(users),
        'users': users
    })
# ==================== SIMPLE FAVICON ROUTE ====================

@app.route('/favicon.ico')
def favicon():
    """Simple favicon route to prevent 404 errors"""
    return '', 204

# ==================== START SERVER ====================

if __name__ == '__main__':
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📁 Results folder: {RESULTS_FOLDER}")
    print(f"👥 User database: {USERS_DB_FILE}")
    print("🚀 Starting Flask server on http://localhost:5005")
    print("🔐 Login page at: http://localhost:5005/")
    print("🆕 Signup page at: http://localhost:5005/signup")
    print("📊 Dashboard at: http://localhost:5005/dashboard")
    print("🔍 Debug paths at: http://localhost:5005/debug-paths")
    app.run(debug=True, host='0.0.0.0', port=5005)