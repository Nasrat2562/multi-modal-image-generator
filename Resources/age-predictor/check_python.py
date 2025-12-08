import sys
print("Python Information:")
print("=" * 40)
print(f"Executable: {sys.executable}")
print(f"Version: {sys.version}")
print("=" * 40)

if "3.11.9" in sys.version:
    print("🎉 SUCCESS! Using Python 3.11.9")
else:
    print(f"❌ WRONG VERSION! Using: {sys.version}")
    print("Please set interpreter to Python 3.11.9")