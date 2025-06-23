"""PandaKinetics"""
__version__ = "0.1.0"

from .types import KineticResults, SimulationResults
from .utils.validation import check_installation, check_gpu_availability

def get_kinetic_simulator():
    from .core.kinetics import KineticSimulator
    return KineticSimulator

KineticSimulator = get_kinetic_simulator()

__all__ = ["KineticSimulator", "KineticResults", "SimulationResults", 
           "check_installation", "check_gpu_availability"]
