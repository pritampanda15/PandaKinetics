#!/usr/bin/env python3
"""Fix circular imports in PandaKinetics"""

from pathlib import Path

def fix_circular_imports():
    print("🔧 Fixing circular import issues...")
    
    # 1. Create types module
    types_content = '''"""Shared data types for PandaKinetics"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch

@dataclass
class KineticResults:
    kon: float
    koff: float  
    residence_time: float
    binding_affinity: float
    kinetic_selectivity: Dict[str, float]
    pathway_analysis: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

@dataclass  
class SimulationResults:
    binding_times: torch.Tensor
    unbinding_times: torch.Tensor
    trajectories: List[torch.Tensor]
    state_populations: torch.Tensor
    transition_counts: torch.Tensor
    total_simulation_time: float
'''
    
    with open('pandakinetics/types.py', 'w') as f:
        f.write(types_content)
    print("✓ Created pandakinetics/types.py")
    
    # 2. Fix main __init__.py
    init_content = '''"""PandaKinetics"""
__version__ = "0.1.0"

from .types import KineticResults, SimulationResults
from .utils.validation import check_installation, check_gpu_availability

def get_kinetic_simulator():
    from .core.kinetics import KineticSimulator
    return KineticSimulator

KineticSimulator = get_kinetic_simulator()

__all__ = ["KineticSimulator", "KineticResults", "SimulationResults", 
           "check_installation", "check_gpu_availability"]
'''
    
    with open('pandakinetics/__init__.py', 'w') as f:
        f.write(init_content)
    print("✓ Fixed pandakinetics/__init__.py")
    
    print("✅ Circular import fix completed!")
    print("\nTest with: python -c 'from pandakinetics.types import KineticResults; print(\"Success!\")'")

if __name__ == "__main__":
    fix_circular_imports()
