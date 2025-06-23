#!/usr/bin/env python3
"""
Quick installation script for PandaKinetics
Run this script to install PandaKinetics with minimal dependencies
"""

import subprocess
import sys
import os

def install():
    """Quick install function"""
    print("🚀 PandaKinetics Quick Install")
    print("=" * 30)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    commands = [
        "python -m pip install --upgrade pip setuptools wheel",
        "python -m pip install numpy scipy pandas matplotlib networkx scikit-learn",
        "python -m pip install tqdm loguru click pydantic",
        "python -m pip install -e ."
    ]
    
    for i, cmd in enumerate(commands, 1):
        print(f"\n[{i}/{len(commands)}] {cmd}")
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"❌ Failed: {cmd}")
            sys.exit(1)
        print("✓ Success")
    
    # Test installation
    print("\n🧪 Testing installation...")
    try:
        import pandakinetics
        print(f"✅ PandaKinetics {pandakinetics.__version__} installed successfully!")
        
        # Test CLI
        result = subprocess.run(f"{sys.executable} -m pandakinetics.cli --version", 
                              shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CLI working correctly")
        
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("• Run: pandakinetics --help")
        print("• Install ML tools: pip install -e '.[ml]'")
        print("• Install GPU support: pip install -e '.[gpu]'")
        print("• Full install: pip install -e '.[full]'")
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        print("\nTry running the setup script manually:")
        print("bash setup_fix.sh")

if __name__ == "__main__":
    install()
