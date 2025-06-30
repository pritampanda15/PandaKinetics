# =============================================================================
# pandakinetics/utils/__init__.py
# =============================================================================

# Import core utilities that should always be available
from .gpu_utils import GPUUtils

# Import validation functions that are needed by main package
try:
    from .validation import check_installation, check_gpu_availability
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    def check_installation():
        return False
    def check_gpu_availability():
        return False

# Import optional utilities with fallbacks
try:
    from .io_handlers import PDBHandler, MoleculeHandler
    IO_HANDLERS_AVAILABLE = True
except ImportError:
    IO_HANDLERS_AVAILABLE = False
    class PDBHandler:
        def __init__(self, *args, **kwargs):
            raise ImportError("PDBHandler requires biotite")
    class MoleculeHandler:
        def __init__(self, *args, **kwargs):
            raise ImportError("MoleculeHandler requires additional dependencies")

try:
    from .validation import StructureValidator, ResultValidator
    VALIDATORS_AVAILABLE = True
except ImportError:
    VALIDATORS_AVAILABLE = False
    class StructureValidator:
        def __init__(self, *args, **kwargs):
            raise ImportError("StructureValidator requires additional dependencies")
    class ResultValidator:
        def __init__(self, *args, **kwargs):
            raise ImportError("ResultValidator requires additional dependencies")

__all__ = ["GPUUtils", "check_gpu_availability", "check_installation",
           "PDBHandler", "MoleculeHandler", "StructureValidator", "ResultValidator"]