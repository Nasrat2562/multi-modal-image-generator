# run.py - SIMPLE VERSION
import os
import sys
import argparse

print("🎨 AI ART TRANSFORMER - Python 3.11.9 Edition")
print("="*60)

# Add app directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5004)
parser.add_argument('--mode', choices=['monet', 'toon', 'both', 'simple'], default='both')
args = parser.parse_args()

try:
    # Import and run Flask app
    from main import app
    
    print(f"\nStarting server on port {args.port}")
    print(f"Mode: {args.mode}")
    print("="*60)
    
    if args.mode in ['monet', 'both']:
        print(f"🎨 Monet: http://localhost:{args.port}/monet")
    if args.mode in ['toon', 'both']:
        print(f"✨ Toon: http://localhost:{args.port}/toon")
    if args.mode == 'simple':
        print(f"🎨 Simple Mode (no AI): http://localhost:{args.port}")
    
    print("="*60)
    print("Server running! Press Ctrl+C to stop")
    print("="*60)
    
    app.run(debug=False, host='0.0.0.0', port=args.port, use_reloader=False)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Run: python setup_311.py")
    print("2. Or install manually: pip install -r requirements_311.txt")
    print("3. Try simple mode: python run.py --mode simple")
    input("\nPress Enter to exit...")