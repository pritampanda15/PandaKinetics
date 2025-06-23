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
