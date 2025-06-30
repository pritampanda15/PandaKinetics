# =============================================================================
# pandakinetics/ai/__init__.py - AI Module Initialization
# =============================================================================

"""
AI and Machine Learning components for PandaKinetics

This module provides neural network models and AI-enhanced sampling methods
for predicting molecular properties and transition barriers.
"""

# Import models with fallback handling
try:
    from .models import (
        TransitionBarrierNet,
        MolecularEncoder,
        EnergyPredictor,
        KineticPredictor,
        create_model,
        get_model_info
    )
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AI models not fully available: {e}")
    MODELS_AVAILABLE = False
    
    # Create dummy classes for compatibility
    class TransitionBarrierNet:
        def __init__(self, *args, **kwargs):
            raise ImportError("TransitionBarrierNet requires torch")
    
    def create_model(*args, **kwargs):
        raise ImportError("Model creation requires torch")
    
    def get_model_info():
        return {"models_available": False}

# Import other AI components with similar fallback
try:
    from .barrier_predictor import BarrierPredictor
    BARRIER_PREDICTOR_AVAILABLE = True
except ImportError:
    BARRIER_PREDICTOR_AVAILABLE = False
    
    class BarrierPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("BarrierPredictor requires additional dependencies")

try:
    from .sampling import EnhancedSampler
    ENHANCED_SAMPLER_AVAILABLE = True
except ImportError:
    ENHANCED_SAMPLER_AVAILABLE = False
    
    class EnhancedSampler:
        def __init__(self, *args, **kwargs):
            raise ImportError("EnhancedSampler requires additional dependencies")

# Import new Boltz-2 inspired modules
try:
    from .boltz_inspired_affinity import (
        BoltzInspiredAffinityPredictor,
        BoltzInspiredAffinityModule,
        PairFormerAttention,
        StructuralEncoder
    )
    BOLTZ_AFFINITY_AVAILABLE = True
except ImportError:
    BOLTZ_AFFINITY_AVAILABLE = False
    
    class BoltzInspiredAffinityPredictor:
        def __init__(self, *args, **kwargs):
            raise ImportError("BoltzInspiredAffinityPredictor requires additional dependencies")
    
    class BoltzInspiredAffinityModule:
        def __init__(self, *args, **kwargs):
            raise ImportError("BoltzInspiredAffinityModule requires additional dependencies")

try:
    from .enhanced_transition_analysis import TransitionStateAnalyzer
    TRANSITION_ANALYSIS_AVAILABLE = True
except ImportError:
    TRANSITION_ANALYSIS_AVAILABLE = False
    
    class TransitionStateAnalyzer:
        def __init__(self, *args, **kwargs):
            raise ImportError("TransitionStateAnalyzer requires additional dependencies")

# Module info
AI_MODULE_INFO = {
    "models_available": MODELS_AVAILABLE,
    "barrier_predictor_available": BARRIER_PREDICTOR_AVAILABLE,
    "enhanced_sampler_available": ENHANCED_SAMPLER_AVAILABLE,
    "boltz_affinity_available": BOLTZ_AFFINITY_AVAILABLE,
    "transition_analysis_available": TRANSITION_ANALYSIS_AVAILABLE
}

__all__ = [
    "TransitionBarrierNet",
    "BarrierPredictor",
    "EnhancedSampler",
    "MolecularEncoder",
    "EnergyPredictor",
    "KineticPredictor",
    "BoltzInspiredAffinityPredictor",
    "BoltzInspiredAffinityModule",
    "PairFormerAttention",
    "StructuralEncoder",
    "TransitionStateAnalyzer",
    "create_model",
    "get_model_info",
    "AI_MODULE_INFO"
]
