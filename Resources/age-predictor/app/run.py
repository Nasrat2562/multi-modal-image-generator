import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🤖 AI Age Verification & Authentication System")
print("=" * 60)
print("Project Structure:")
print(f"• Working directory: {os.getcwd()}")
print(f"• Script location: {__file__}")
print("=" * 60)
print("Features:")
print("• Traditional username/password login")
print("• Face recognition login with age verification")
print("• Real-time camera age detection")
print("• Must be 18+ to proceed")
print("• Standalone age predictor")
print("• User Signup with local authentication (No Firebase!)")
print("• Automatic dashboard redirection")
print("=" * 60)

try:
    # Test imports
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
    
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
    
    import flask
    print(f"✓ Flask: {flask.__version__}")
    
    # Check for built-in modules
    import json
    print("✓ JSON (built-in)")
    
    import hashlib
    print("✓ Hashlib (built-in for password security)")
    
    from main import app
    
    print("\n🔍 Checking configuration...")
    
    # Check template folder
    template_path = os.path.join(os.path.dirname(__file__), 'templates')
    if os.path.exists(template_path):
        print(f"✓ Templates folder found: {template_path}")
        files = os.listdir(template_path)
        print(f"  Files in templates: {', '.join(files)}")
    else:
        print(f"❌ Templates folder NOT found: {template_path}")
    
    # Check signup.html specifically
    signup_path = os.path.join(template_path, 'signup.html')
    if os.path.exists(signup_path):
        print(f"✅ signup.html found: {signup_path}")
    else:
        print(f"❌ signup.html NOT found: {signup_path}")
    
    print("\n✅ All systems ready!")
    print("📍 Starting server at: http://localhost:5005")
    print("=" * 50)
    print("📱 Access URLs:")
    print("🔐 Login:      http://localhost:5005/")
    print("🆕 Signup:     http://localhost:5005/signup")
    print("📊 Dashboard:  http://localhost:5005/dashboard")
    print("🎯 Age Tool:   http://localhost:5005/age-predictor")
    print("🔍 Debug:      http://localhost:5005/debug-paths")
    print("=" * 50)
    print("\n🚀 Starting Flask server...")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5005)
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\n📦 Install required packages:")
    print("pip install numpy opencv-python flask pillow")
    input("\nPress Enter to exit...")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    input("\nPress Enter to exit...")