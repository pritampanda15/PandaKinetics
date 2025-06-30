
# PandaKinetics

Multi-Scale Structure-Kinetics Simulator for Drug Design

## Overview

PandaKinetics is a GPU-accelerated toolkit for predicting drug binding kinetics, residence times, and kinetic selectivity using AI-enhanced molecular dynamics simulations.

## Features

- **Enhanced Docking**: Multi-site ensemble docking with conformational diversity
- **AI-Powered Barriers**: Machine learning prediction of transition state energies
- **Kinetic Monte Carlo**: GPU-accelerated simulation of binding/unbinding kinetics
- **Kinetic Selectivity**: Prediction of selectivity based on residence times
- **Comprehensive Analysis**: Statistical analysis and visualization tools

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU
- CUDA Toolkit 12.0+

### Install from PyPI
```bash
pip install pandakinetics
```

### Install from source
```bash
git clone https://github.com/pandakinetics/pandakinetics.git
cd pandakinetics
pip install -e .
```

## Quick Start

### Basic Usage
```python
from pandakinetics import KineticSimulator

# Initialize simulator
simulator = KineticSimulator(
    temperature=310.0,  # Physiological temperature
    n_replicas=16,      # Parallel simulations
)

# Predict kinetics
results = simulator.predict_kinetics(
    protein_pdb="1ABC",  # PDB ID or file path
    ligand_smiles="CCO"  # Ethanol as example
)

# Access results
print(f"Association rate: {results.kon:.2e} M⁻¹s⁻¹")
print(f"Dissociation rate: {results.koff:.2e} s⁻¹")
print(f"Residence time: {results.residence_time:.2e} s")
```

### Command Line Interface
```bash
# Predict kinetics
pandakinetics predict --protein 1ABC --ligand "CCO" --output results/

# Visualize results
pandakinetics visualize --results-file results/kinetic_results.json

# Benchmark GPU
pandakinetics benchmark
```

### Advanced Usage
```python
from pandakinetics import KineticSimulator, BarrierPredictor

# Custom barrier predictor
barrier_predictor = BarrierPredictor(
    model_path="custom_model.pt",
    hidden_dim=256
)

# Advanced simulator configuration
simulator = KineticSimulator(
    temperature=310.0,
    n_replicas=32,
    max_simulation_time=1e-2,  # 10 ms
    barrier_predictor=barrier_predictor
)

# Predict with custom binding sites
binding_sites = [
    {'center': [10.0, 15.0, 20.0], 'radius': 10.0},
    {'center': [30.0, 25.0, 10.0], 'radius': 8.0}
]

results = simulator.predict_kinetics(
    protein_pdb="protein.pdb",
    ligand_smiles="complex_molecule_smiles",
    binding_sites=binding_sites,
    reference_ligands=["reference_smiles1", "reference_smiles2"]
)

# Analyze selectivity
for ref_ligand, selectivity in results.kinetic_selectivity.items():
    print(f"Selectivity vs {ref_ligand}: {selectivity:.2f}")
```

## Key Components

### 1. Enhanced Docking Engine
- Multi-conformer generation
- Ensemble docking across multiple binding sites
- Pose clustering and filtering
- GPU-accelerated scoring

### 2. AI Barrier Predictor
- E(3)-equivariant neural networks
- Transition state energy prediction
- Physics-informed constraints
- Pre-trained on MD simulation data

### 3. Kinetic Monte Carlo Simulator
- Parallel replica simulations
- Transition rate calculations
- Binding/unbinding event detection
- Statistical analysis

### 4. Analysis and Visualization
- Kinetic parameter estimation
- Confidence interval calculation
- Pathway analysis
- Interactive plotting

## Performance

PandaKinetics is optimized for GPU acceleration:

- **Docking**: 100+ poses in seconds
- **Barrier Prediction**: 1000+ transitions per second
- **Monte Carlo**: Million+ steps per second (parallel)
- **Memory**: Efficient GPU memory management

## Applications

### Drug Discovery
- Lead optimization for residence time
- Kinetic selectivity design
- ADMET property prediction

### Research
- Binding mechanism elucidation
- Allosteric pathway analysis
- Structure-kinetics relationships

## Validation

PandaKinetics has been validated against:
- Experimental SPR data
- Literature kinetic parameters
- Benchmark datasets

## Citation

If you use PandaKinetics in your research, please cite:

```
@article{pandakinetics2025,
  title={PandaKinetics: Multi-Scale Structure-Kinetics Simulator for Drug Design},
  author={Pritam Kumar Panda},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: https://pritampanda15.readthedocs.io
- Issues: https://github.com/pritampanda15/pandakinetics/issues
- Discussions: https://github.com/pritampanda15/pandakinetics/discussions

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Acknowledgments

- Built on PyTorch, RDKit, OpenMM, and Biotite
- Inspired by advances in AI-accelerated MD