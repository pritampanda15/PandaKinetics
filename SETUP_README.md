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
