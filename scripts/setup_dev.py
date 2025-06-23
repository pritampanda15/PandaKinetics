#!/usr/bin/env python3
"""
Development environment setup script for PandaKinetics
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run shell command and handle errors"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True)
    
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        sys.exit(1)
    
    return result

def setup_environment():
    """Setup development environment"""
    print("Setting up development environment...")
    
    # Install package in development mode
    run_command(f"{sys.executable} -m pip install -e '.[dev]'")
    
    # Create necessary directories
    dirs = ["results", "logs", "temp", "tests/data"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("Development environment setup complete!")

def main():
    """Main setup function"""
    print("PandaKinetics Development Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher required")
        sys.exit(1)
    
    setup_environment()
    
    # Test installation
    try:
        import pandakinetics
        print(f"\n✓ PandaKinetics {pandakinetics.__version__} installed successfully!")
    except ImportError as e:
        print(f"\n✗ Installation failed: {e}")

if __name__ == "__main__":
    main()
