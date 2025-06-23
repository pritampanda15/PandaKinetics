#!/bin/bash

# =============================================================================
# PandaKinetics Setup Fix
# =============================================================================

echo "🔧 Fixing PandaKinetics Installation Issues"
echo "============================================"

# Step 1: Fix virtual environment pip issue
echo "Step 1: Fixing virtual environment..."

# Deactivate current environment if active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Deactivating current virtual environment..."
    deactivate 2>/dev/null || true
fi

# Remove problematic virtual environment
if [ -d ".venv" ]; then
    echo "Removing corrupted virtual environment..."
    rm -rf .venv
fi

# Create fresh virtual environment
echo "Creating fresh virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip in virtual environment
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Step 2: Create correct file structure
echo "Step 2: Creating correct file structure..."

# Create necessary directories
mkdir -p scripts
mkdir -p tests/data
mkdir -p pandakinetics/data
mkdir -p docs/source
mkdir -p examples
mkdir -p docker

# Step 3: Move and create correct files
echo "Step 3: Setting up configuration files..."

# Create the correct setup.py (minimal)
cat > setup.py << 'EOF'
#!/usr/bin/env python3
"""
Legacy setup.py for PandaKinetics
Modern configuration is in pyproject.toml
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
EOF

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pandakinetics"
version = "0.1.0"
description = "Multi-Scale Structure-Kinetics Simulator for Drug Design"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "PandaKinetics Team", email = "contact@pandakinetics.org"}
]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "matplotlib>=3.5.0",
    "networkx>=2.6",
    "tqdm>=4.60.0",
    "loguru>=0.6.0",
    "click>=8.0.0",
    "pydantic>=1.8.0",
    "scikit-learn>=1.0.0",
]

[project.optional-dependencies]
ml = [
    "torch>=1.12.0",
    "torchvision>=0.13.0",
]
gpu = [
    "cupy-cuda11x>=10.0.0; sys_platform=='linux'",
]
chem = [
    "rdkit>=2022.3.1",
    "biotite>=0.36.0",
    "mdtraj>=1.9.6",
]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.950",
]
full = ["pandakinetics[ml,gpu,chem]"]

[project.scripts]
pandakinetics = "pandakinetics.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["pandakinetics*"]
exclude = ["tests*"]
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.5.0
networkx>=2.6
scikit-learn>=1.0.0
tqdm>=4.60.0
loguru>=0.6.0
click>=8.0.0
pydantic>=1.8.0
EOF

# Create basic package structure if it doesn't exist
if [ ! -f "pandakinetics/__init__.py" ]; then
    echo "Creating basic package structure..."
    
    mkdir -p pandakinetics/core
    mkdir -p pandakinetics/utils
    
    # Create basic __init__.py
    cat > pandakinetics/__init__.py << 'EOF'
"""
PandaKinetics: Multi-Scale Structure-Kinetics Simulator for Drug Design
"""

__version__ = "0.1.0"
__author__ = "PandaKinetics Team"

# Basic imports for testing
from .utils.validation import check_installation

__all__ = ["__version__", "check_installation"]
EOF

    # Create basic utils module
    mkdir -p pandakinetics/utils
    cat > pandakinetics/utils/__init__.py << 'EOF'
"""Utility functions for PandaKinetics"""

from .validation import check_installation, check_gpu_availability

__all__ = ["check_installation", "check_gpu_availability"]
EOF

    # Create basic validation module
    cat > pandakinetics/utils/validation.py << 'EOF'
"""Validation utilities for PandaKinetics"""

def check_installation():
    """Check if PandaKinetics is properly installed"""
    try:
        import numpy
        import scipy
        import pandas
        import matplotlib
        print("✓ Core dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        return False

def check_gpu_availability():
    """Check GPU availability"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
EOF

    # Create basic CLI module
    cat > pandakinetics/cli.py << 'EOF'
"""Command-line interface for PandaKinetics"""

import click

@click.command()
@click.option('--version', is_flag=True, help='Show version')
def main(version):
    """PandaKinetics: Multi-Scale Structure-Kinetics Simulator"""
    if version:
        from . import __version__
        click.echo(f"PandaKinetics version {__version__}")
    else:
        click.echo("PandaKinetics is ready!")
        click.echo("Use --help for more options")

if __name__ == "__main__":
    main()
EOF
fi

# Create the correct development setup script
cat > scripts/setup_dev.py << 'EOF'
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
EOF

# Make setup script executable
chmod +x scripts/setup_dev.py

# Step 4: Install the package
echo "Step 4: Installing PandaKinetics..."

# Install core dependencies first
python -m pip install -r requirements.txt

# Install package in development mode
python -m pip install -e .

# Step 5: Test installation
echo "Step 5: Testing installation..."

# Test basic import
python -c "import pandakinetics; print(f'✓ PandaKinetics {pandakinetics.__version__} imported successfully')"

# Test CLI
python -m pandakinetics.cli --version

echo ""
echo "🎉 Installation Fixed Successfully!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Test the installation:"
echo "   python -c 'import pandakinetics; pandakinetics.check_installation()'"
echo ""
echo "2. Install additional components:"
echo "   pip install -e '.[ml]'     # Machine learning"
echo "   pip install -e '.[gpu]'    # GPU support" 
echo "   pip install -e '.[chem]'   # Chemistry tools"
echo "   pip install -e '.[full]'   # Everything"
echo ""
echo "3. Run development setup:"
echo "   python scripts/setup_dev.py"
echo ""
echo "4. Start developing:"
echo "   pandakinetics --help"

# =============================================================================
# quick_install.py - Standalone installation script
# =============================================================================

cat > quick_install.py << 'EOF'
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
EOF

# Make quick install executable
chmod +x quick_install.py

echo ""
echo "📄 Files created:"
echo "  ✓ setup.py (corrected)"
echo "  ✓ pyproject.toml (modern config)"
echo "  ✓ requirements.txt (core deps)"
echo "  ✓ scripts/setup_dev.py (dev setup)"
echo "  ✓ quick_install.py (standalone installer)"
echo "  ✓ Basic package structure"
echo ""

# =============================================================================
# Alternative Manual Steps
# =============================================================================

cat > INSTALLATION_STEPS.md << 'EOF'
# Manual Installation Steps

If the automated script fails, follow these manual steps:

## 1. Fix Virtual Environment

```bash
# Remove corrupted venv
rm -rf .venv

# Create fresh venv
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate  # Windows

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install Dependencies

```bash
# Install core dependencies
pip install numpy scipy pandas matplotlib networkx scikit-learn
pip install tqdm loguru click pydantic

# Install package
pip install -e .
```

## 3. Test Installation

```bash
# Test import
python -c "import pandakinetics; print('Success!')"

# Test CLI
pandakinetics --version
```

## 4. Install Additional Components (Optional)

```bash
# Machine Learning
pip install torch torchvision

# GPU Support (if CUDA available)
pip install cupy-cuda11x

# Chemistry Tools
pip install rdkit biotite mdtraj

# All at once
pip install -e '.[full]'
```

## 5. Development Setup (Optional)

```bash
# Development tools
pip install pytest black isort flake8 mypy

# Pre-commit hooks (if contributing)
pip install pre-commit
pre-commit install
```

## Troubleshooting

### Issue: "No module named 'pip'"
**Solution:** Recreate virtual environment
```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Issue: "CUDA not found"
**Solution:** Install CPU-only versions first
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue: "RDKit installation fails"
**Solution:** Use conda for RDKit
```bash
conda install -c conda-forge rdkit
# Then continue with pip for other packages
```

### Issue: Permission denied
**Solution:** Use user installation or virtual environment
```bash
pip install --user pandakinetics
# OR ensure you're in a virtual environment
```

## Verification Commands

```bash
# Check installation
python -c "
import pandakinetics
print(f'PandaKinetics {pandakinetics.__version__}')
pandakinetics.check_installation()
"

# Check GPU support
python -c "
from pandakinetics.utils import check_gpu_availability
print(f'GPU available: {check_gpu_availability()}')
"

# Check CLI
pandakinetics --help
```
EOF

# =============================================================================
# Create README for setup
# =============================================================================

cat > SETUP_README.md << 'EOF'
# PandaKinetics Setup Guide

## Quick Start (Recommended)

### Option 1: Automated Fix
```bash
# Run the fix script
bash setup_fix.sh
```

### Option 2: Quick Install
```bash
# Run quick installer
python quick_install.py
```

### Option 3: Manual Steps
See `INSTALLATION_STEPS.md` for detailed manual installation.

## What This Fixes

1. **Virtual Environment Issues**: Creates fresh .venv with working pip
2. **File Structure**: Sets up correct package structure
3. **Dependencies**: Installs core dependencies in correct order
4. **Configuration**: Creates proper pyproject.toml and setup.py

## After Installation

### Basic Test
```bash
python -c "import pandakinetics; print('✓ Working!')"
pandakinetics --version
```

### Install Additional Features
```bash
# Machine Learning
pip install -e '.[ml]'

# GPU Support
pip install -e '.[gpu]'

# Chemistry Tools  
pip install -e '.[chem]'

# Everything
pip install -e '.[full]'
```

### Development Mode
```bash
python scripts/setup_dev.py
```

## File Structure Created

```
pandakinetics/
├── pandakinetics/          # Main package
│   ├── __init__.py
│   ├── cli.py
│   └── utils/
│       ├── __init__.py
│       └── validation.py
├── scripts/
│   └── setup_dev.py
├── setup.py               # Fixed
├── pyproject.toml         # Proper config
├── requirements.txt       # Core deps
└── quick_install.py      # Standalone installer
```

## Support

If you still have issues:
1. Check `INSTALLATION_STEPS.md` for manual steps
2. Ensure Python 3.8+ is installed
3. Try in a fresh virtual environment
4. Report issues with full error messages

## Next Steps

1. **Test Installation**: `python -c "import pandakinetics"`
2. **Install ML/GPU**: `pip install -e '.[full]'`
3. **Run Examples**: Check examples/ directory (when available)
4. **Contribute**: Follow CONTRIBUTING.md guidelines
EOF

echo "📚 Documentation created:"
echo "  ✓ INSTALLATION_STEPS.md (manual steps)"
echo "  ✓ SETUP_README.md (overview)"
echo ""
echo "🔧 To fix your installation now, run:"
echo "   bash setup_fix.sh"
echo ""
echo "🚀 Or for quick install:"
echo "   python quick_install.py"