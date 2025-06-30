"""PandaKinetics"""
__version__ = "0.1.0"

from .types import KineticResults, SimulationResults
from .utils.validation import check_installation, check_gpu_availability

def get_kinetic_simulator():
    """Get KineticSimulator class (lazy import to avoid torch dependency for CLI)"""
    try:
        from .core.kinetics import KineticSimulator
        return KineticSimulator
    except ImportError as e:
        # Return a dummy class for CLI compatibility
        class KineticSimulator:
            def __init__(self, *args, **kwargs):
                raise ImportError(f"KineticSimulator requires additional dependencies: {e}")
        return KineticSimulator

# Only import KineticSimulator when actually needed
KineticSimulator = None

def __getattr__(name):
    """Lazy attribute access for KineticSimulator"""
    global KineticSimulator
    if name == "KineticSimulator":
        if KineticSimulator is None:
            KineticSimulator = get_kinetic_simulator()
        return KineticSimulator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["KineticSimulator", "KineticResults", "SimulationResults", 
           "check_installation", "check_gpu_availability"]
