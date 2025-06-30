"""Shared data types for PandaKinetics"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Conditional torch import for CLI compatibility
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy torch for type hints
    class torch:
        class Tensor:
            pass

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
