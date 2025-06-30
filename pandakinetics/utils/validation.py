# =============================================================================
# Complete pandakinetics/utils/validation.py with missing functions
# =============================================================================

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

# Conditional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy torch
    class torch:
        @staticmethod
        def cuda():
            class CUDA:
                @staticmethod
                def is_available():
                    return False
            return CUDA()

# Import with fallbacks
try:
    import biotite.structure as struc
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False
    logger.warning("Biotite not available - protein structure validation limited")

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - ligand validation limited")

# Import shared types with fallback
try:
    from ..types import KineticResults, SimulationResults
except ImportError:
    # Create dummy classes for compatibility
    class KineticResults:
        def __init__(self):
            self.kon = 0
            self.koff = 0
            self.residence_time = 0
            self.binding_affinity = 0
            self.kinetic_selectivity = {}
            self.pathway_analysis = {}
            self.confidence_intervals = {}
    
    class SimulationResults:
        def __init__(self):
            self.binding_times = torch.tensor([])
            self.unbinding_times = torch.tensor([])
            self.trajectories = []
            self.state_populations = torch.tensor([])
            self.transition_counts = torch.tensor([])
            self.total_simulation_time = 0


# =============================================================================
# MISSING FUNCTIONS - Add these to fix the import error
# =============================================================================

def check_installation() -> bool:
    """
    Check if PandaKinetics is properly installed with all dependencies
    
    Returns:
        bool: True if installation is complete, False otherwise
    """
    logger.info("Checking PandaKinetics installation...")
    
    issues = []
    warnings = []
    
    # Check core Python packages
    core_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'scipy': 'SciPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'networkx': 'NetworkX',
        'loguru': 'Loguru',
        'click': 'Click',
        'tqdm': 'TQDM'
    }
    
    for package, name in core_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {name} available")
        except ImportError:
            issues.append(f"✗ {name} not available - install with: pip install {package}")
    
    # Check optional packages
    optional_packages = {
        'cupy': 'CuPy (GPU acceleration)',
        'rdkit': 'RDKit (chemistry)',
        'biotite': 'Biotite (structure)',
        'mdtraj': 'MDTraj (trajectories)',
        'torch_geometric': 'PyTorch Geometric (graph networks)',
        'e3nn': 'E3NN (equivariant networks)',
        'openmm': 'OpenMM (MD simulations)'
    }
    
    for package, name in optional_packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {name} available")
        except ImportError:
            warnings.append(f"○ {name} not available (optional)")
    
    # Check PandaKinetics modules
    try:
        from .. import types
        logger.info("✓ PandaKinetics types module available")
    except ImportError as e:
        issues.append(f"✗ PandaKinetics types module error: {e}")
    
    try:
        from ..core import kinetics
        logger.info("✓ PandaKinetics core module available")
    except ImportError as e:
        warnings.append(f"○ PandaKinetics core module warning: {e}")
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    if gpu_available:
        logger.info("✓ GPU acceleration available")
    else:
        warnings.append("○ GPU acceleration not available (will use CPU)")
    
    # Print summary
    if issues:
        logger.error("Installation issues found:")
        for issue in issues:
            logger.error(f"  {issue}")
        return False
    else:
        logger.info("✓ Core installation is complete")
        
        if warnings:
            logger.info("Optional components:")
            for warning in warnings:
                logger.info(f"  {warning}")
        
        return True


def check_gpu_availability() -> bool:
    """
    Check if GPU acceleration is available
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    gpu_available = False
    
    # Check PyTorch CUDA
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"✓ PyTorch CUDA available: {device_name} ({gpu_count} devices)")
                gpu_available = True
            else:
                logger.info("○ PyTorch CUDA not available")
        except Exception as e:
            logger.warning(f"○ PyTorch CUDA check failed: {e}")
    else:
        logger.info("○ PyTorch not available")
    
    # Check CuPy
    try:
        import cupy as cp
        cp.cuda.Device(0).use()
        test_array = cp.array([1, 2, 3])
        logger.info("✓ CuPy available")
        gpu_available = True
    except ImportError:
        logger.info("○ CuPy not available")
    except Exception:
        logger.info("○ CuPy available but GPU access failed")
    
    # Check NVIDIA drivers
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✓ NVIDIA drivers available")
        else:
            logger.info("○ NVIDIA drivers not accessible")
    except FileNotFoundError:
        logger.info("○ nvidia-smi not found")
    except Exception:
        logger.info("○ Could not check NVIDIA drivers")
    
    return gpu_available


def validate_environment() -> Dict[str, Any]:
    """
    Comprehensive environment validation
    
    Returns:
        Dict with validation results
    """
    validation = {
        'core_installation': check_installation(),
        'gpu_available': check_gpu_availability(),
        'issues': [],
        'warnings': [],
        'recommendations': []
    }
    
    # Check Python version
    import sys
    python_version = sys.version_info
    if python_version < (3, 8):
        validation['issues'].append(f"Python {python_version.major}.{python_version.minor} is too old (need 3.8+)")
    elif python_version >= (3, 12):
        validation['warnings'].append(f"Python {python_version.major}.{python_version.minor} is very new - some packages may not be compatible")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            validation['warnings'].append(f"Low system memory: {memory_gb:.1f} GB (recommend 8+ GB)")
        elif memory_gb >= 16:
            logger.info(f"✓ Sufficient memory: {memory_gb:.1f} GB")
    except ImportError:
        validation['warnings'].append("Could not check system memory")
    
    # Add recommendations
    if not validation['core_installation']:
        validation['recommendations'].append("Run: pip install pandakinetics[full]")
    
    if not validation['gpu_available']:
        validation['recommendations'].extend([
            "For GPU acceleration:",
            "  1. Install CUDA toolkit",
            "  2. Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118",
            "  3. Install CuPy: pip install cupy-cuda11x"
        ])
    
    return validation


def check_dependencies() -> Dict[str, bool]:
    """
    Check individual dependency availability
    
    Returns:
        Dict mapping package names to availability status
    """
    dependencies = {}
    
    # Core dependencies
    core_deps = ['torch', 'numpy', 'scipy', 'pandas', 'matplotlib', 'networkx', 'loguru', 'click', 'tqdm']
    for dep in core_deps:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    # Optional dependencies
    optional_deps = ['cupy', 'rdkit', 'biotite', 'mdtraj', 'torch_geometric', 'e3nn', 'openmm']
    for dep in optional_deps:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return dependencies


# =============================================================================
# EXISTING VALIDATOR CLASSES (keeping the original structure)
# =============================================================================

class StructureValidator:
    """Validator for molecular structures and inputs"""
    
    @staticmethod
    def validate_protein_structure(structure) -> Dict[str, Any]:
        """Validate protein structure quality"""
        
        if not BIOTITE_AVAILABLE:
            return {
                'valid': False,
                'warnings': [],
                'errors': ['Biotite not available - cannot validate protein structure']
            }
        
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check for minimum number of atoms
        if len(structure) < 100:
            validation['errors'].append("Structure too small (< 100 atoms)")
            validation['valid'] = False
        
        # Check for missing backbone atoms
        backbone_atoms = ["N", "CA", "C", "O"]
        for atom_name in backbone_atoms:
            if not np.any(structure.atom_name == atom_name):
                validation['errors'].append(f"Missing backbone atoms: {atom_name}")
                validation['valid'] = False
        
        # Check for unusual residue names
        standard_residues = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
        ]
        
        unusual_residues = set(structure.res_name) - set(standard_residues)
        if unusual_residues:
            validation['warnings'].append(f"Unusual residues found: {unusual_residues}")
        
        # Check for chain breaks
        ca_atoms = structure[structure.atom_name == "CA"]
        if len(ca_atoms) > 1:
            distances = np.linalg.norm(np.diff(ca_atoms.coord, axis=0), axis=1)
            long_distances = distances > 5.0  # Angstroms
            if np.any(long_distances):
                validation['warnings'].append(f"Potential chain breaks detected: {np.sum(long_distances)} gaps")
        
        # Check coordinate quality
        coords = structure.coord
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            validation['errors'].append("Invalid coordinates (NaN or Inf)")
            validation['valid'] = False
        
        # Check for reasonable coordinate ranges
        coord_ranges = np.ptp(coords, axis=0)  # Peak-to-peak
        if np.any(coord_ranges > 200.0):  # Very large structures
            validation['warnings'].append("Very large coordinate range (> 200 Å)")
        
        return validation
    
    @staticmethod
    def validate_ligand_molecule(mol) -> Dict[str, Any]:
        """Validate ligand molecule"""
        
        if not RDKIT_AVAILABLE:
            return {
                'valid': False,
                'warnings': [],
                'errors': ['RDKit not available - cannot validate ligand molecule']
            }
        
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        if mol is None:
            validation['errors'].append("Invalid molecule (None)")
            validation['valid'] = False
            return validation
        
        # Check atom count
        n_atoms = mol.GetNumAtoms()
        if n_atoms < 5:
            validation['warnings'].append("Very small molecule (< 5 atoms)")
        elif n_atoms > 100:
            validation['warnings'].append("Very large molecule (> 100 atoms)")
        
        # Check for 3D coordinates
        try:
            conf = mol.GetConformer()
            positions = conf.GetPositions()
            
            # Check if coordinates are 3D (not all zeros in z)
            if np.allclose(positions[:, 2], 0):
                validation['warnings'].append("Molecule appears to be 2D (z-coordinates are zero)")
            
        except ValueError:
            validation['errors'].append("No conformer found - 3D coordinates required")
            validation['valid'] = False
        
        # Check for disconnected fragments
        if RDKIT_AVAILABLE:
            fragments = Chem.GetMolFrags(mol)
            if len(fragments) > 1:
                validation['warnings'].append(f"Molecule has {len(fragments)} disconnected fragments")
        
        # Check for unusual valences
        try:
            if RDKIT_AVAILABLE:
                Chem.SanitizeMol(mol)
        except Exception as e:
            validation['warnings'].append(f"Sanitization warning: {e}")
        
        return validation
    
    @staticmethod
    def validate_binding_sites(
        binding_sites: List[Dict], 
        structure
    ) -> Dict[str, Any]:
        """Validate binding site definitions"""
        
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        if not binding_sites:
            validation['warnings'].append("No binding sites provided")
            return validation
        
        if not BIOTITE_AVAILABLE:
            validation['warnings'].append("Biotite not available - limited binding site validation")
            return validation
        
        structure_coords = structure.coord
        structure_center = np.mean(structure_coords, axis=0)
        structure_radius = np.max(np.linalg.norm(structure_coords - structure_center, axis=1))
        
        for i, site in enumerate(binding_sites):
            # Check required fields
            if 'center' not in site:
                validation['errors'].append(f"Binding site {i}: missing 'center' field")
                validation['valid'] = False
                continue
            
            if 'radius' not in site:
                validation['warnings'].append(f"Binding site {i}: missing 'radius' field, using default")
                site['radius'] = 10.0
            
            # Check center coordinates
            center = np.array(site['center'])
            if len(center) != 3:
                validation['errors'].append(f"Binding site {i}: center must be 3D coordinates")
                validation['valid'] = False
                continue
            
            # Check if site is within structure bounds
            distance_to_structure = np.linalg.norm(center - structure_center)
            if distance_to_structure > structure_radius + site['radius']:
                validation['warnings'].append(
                    f"Binding site {i}: appears to be outside protein structure"
                )
            
            # Check radius
            if site['radius'] <= 0:
                validation['errors'].append(f"Binding site {i}: radius must be positive")
                validation['valid'] = False
            elif site['radius'] > 50.0:
                validation['warnings'].append(f"Binding site {i}: very large radius (> 50 Å)")
        
        return validation


class ResultValidator:
    """Validator for simulation results and kinetic parameters"""
    
    @staticmethod
    def validate_kinetic_results(results: KineticResults) -> Dict[str, Any]:
        """Validate kinetic simulation results"""
        
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'quality_score': 0.0
        }
        
        # Check rate values
        if results.kon <= 0:
            validation['warnings'].append("Association rate (kon) is zero or negative")
        elif results.kon > 1e10:  # Very fast association
            validation['warnings'].append("Association rate (kon) is unusually high")
        
        if results.koff <= 0:
            validation['warnings'].append("Dissociation rate (koff) is zero or negative")
        elif results.koff > 1e6:  # Very fast dissociation
            validation['warnings'].append("Dissociation rate (koff) is unusually high")
        
        # Check thermodynamic consistency
        if results.binding_affinity <= 0:
            validation['errors'].append("Binding affinity (Kd) must be positive")
            validation['valid'] = False
        
        # Check residence time
        if results.residence_time < 1e-9:  # Less than 1 ns
            validation['warnings'].append("Residence time is very short (< 1 ns)")
        elif results.residence_time > 1e6:  # More than ~11 days
            validation['warnings'].append("Residence time is very long (> 11 days)")
        
        # Check confidence intervals
        if hasattr(results, 'confidence_intervals') and results.confidence_intervals:
            for param, (lower, upper) in results.confidence_intervals.items():
                if lower >= upper:
                    validation['errors'].append(f"Invalid confidence interval for {param}: [{lower}, {upper}]")
                    validation['valid'] = False
                
                # Check for reasonable interval widths
                if param in ['kon_ci', 'koff_ci']:
                    width_ratio = (upper - lower) / (lower + 1e-10)
                    if width_ratio > 10.0:  # Very wide intervals
                        validation['warnings'].append(f"Very wide confidence interval for {param}")
        
        # Calculate quality score
        quality_factors = []
        
        # Factor 1: Rate reasonableness
        if 1e3 <= results.kon <= 1e9 and 1e-6 <= results.koff <= 1e3:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.5)
        
        # Factor 2: Confidence interval availability
        if hasattr(results, 'confidence_intervals') and results.confidence_intervals:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.0)
        
        # Factor 3: Selectivity data
        if results.kinetic_selectivity:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.0)
        
        validation['quality_score'] = np.mean(quality_factors)
        
        return validation
    
    @staticmethod
    def validate_simulation_results(results: SimulationResults) -> Dict[str, Any]:
        """Validate Monte Carlo simulation results"""
        
        validation = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'convergence_score': 0.0
        }
        
        # Check event counts
        n_binding = len(results.binding_times)
        n_unbinding = len(results.unbinding_times)
        
        if n_binding == 0:
            validation['warnings'].append("No binding events observed")
        elif n_binding < 10:
            validation['warnings'].append(f"Low number of binding events ({n_binding})")
        
        if n_unbinding == 0:
            validation['warnings'].append("No unbinding events observed")
        elif n_unbinding < 10:
            validation['warnings'].append(f"Low number of unbinding events ({n_unbinding})")
        
        # Check for negative times
        if n_binding > 0 and torch.any(results.binding_times < 0):
            validation['errors'].append("Negative binding times found")
            validation['valid'] = False
        
        if n_unbinding > 0 and torch.any(results.unbinding_times < 0):
            validation['errors'].append("Negative unbinding times found")
            validation['valid'] = False
        
        # Check trajectory consistency
        n_trajectories = len(results.trajectories)
        if n_trajectories == 0:
            validation['warnings'].append("No trajectories recorded")
        else:
            empty_trajectories = sum(1 for traj in results.trajectories if len(traj) == 0)
            if empty_trajectories == n_trajectories:
                validation['errors'].append("All trajectories are empty")
                validation['valid'] = False
            elif empty_trajectories > n_trajectories * 0.5:
                validation['warnings'].append(f"Many empty trajectories ({empty_trajectories}/{n_trajectories})")
        
        # Check state populations
        if torch.sum(results.state_populations) == 0:
            validation['errors'].append("No state populations recorded")
            validation['valid'] = False
        else:
            # Check for population concentration (poor sampling)
            max_pop = torch.max(results.state_populations)
            total_pop = torch.sum(results.state_populations)
            concentration = max_pop / total_pop
            
            if concentration > 0.9:
                validation['warnings'].append("Population highly concentrated in one state")
        
        # Check transition matrix
        total_transitions = torch.sum(results.transition_counts)
        if total_transitions == 0:
            validation['errors'].append("No transitions recorded")
            validation['valid'] = False
        else:
            # Check for transition diversity
            active_transitions = torch.sum(results.transition_counts > 0)
            total_possible = results.transition_counts.numel()
            diversity = active_transitions.float() / total_possible
            
            if diversity < 0.01:  # Less than 1% of transitions used
                validation['warnings'].append("Very low transition diversity")
        
        # Calculate convergence score
        convergence_factors = []
        
        # Factor 1: Event counts
        if n_binding >= 20 and n_unbinding >= 20:
            convergence_factors.append(1.0)
        elif n_binding >= 10 and n_unbinding >= 10:
            convergence_factors.append(0.7)
        elif n_binding >= 5 and n_unbinding >= 5:
            convergence_factors.append(0.4)
        else:
            convergence_factors.append(0.0)
        
        # Factor 2: Trajectory quality
        if n_trajectories > 0:
            avg_traj_length = np.mean([len(traj) for traj in results.trajectories if len(traj) > 0])
            if avg_traj_length >= 100:
                convergence_factors.append(1.0)
            elif avg_traj_length >= 50:
                convergence_factors.append(0.7)
            elif avg_traj_length >= 20:
                convergence_factors.append(0.4)
            else:
                convergence_factors.append(0.0)
        else:
            convergence_factors.append(0.0)
        
        # Factor 3: State sampling
        if torch.sum(results.state_populations) > 0:
            entropy = -torch.sum(results.state_populations * torch.log(results.state_populations + 1e-10))
            max_entropy = np.log(len(results.state_populations))
            normalized_entropy = entropy / max_entropy
            convergence_factors.append(min(normalized_entropy.item(), 1.0))
        else:
            convergence_factors.append(0.0)
        
        validation['convergence_score'] = np.mean(convergence_factors)
        
        return validation
    
    @staticmethod
    def validate_network_connectivity(adjacency_matrix: torch.Tensor) -> Dict[str, Any]:
        """Validate transition network connectivity"""
        
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        n_states = adjacency_matrix.shape[0]
        
        # Check matrix properties
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            validation['errors'].append("Adjacency matrix must be square")
            validation['valid'] = False
            return validation
        
        # Check symmetry
        if not torch.allclose(adjacency_matrix, adjacency_matrix.T, atol=1e-6):
            validation['warnings'].append("Adjacency matrix is not symmetric")
        
        # Check for self-connections
        if torch.any(torch.diag(adjacency_matrix) > 0):
            validation['warnings'].append("Self-connections found in adjacency matrix")
        
        # Check connectivity
        adj_np = adjacency_matrix.cpu().numpy()
        
        # Find connected components
        visited = np.zeros(n_states, dtype=bool)
        components = []
        
        for i in range(n_states):
            if not visited[i]:
                component = []
                stack = [i]
                
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        component.append(node)
                        
                        # Add neighbors
                        neighbors = np.where(adj_np[node] > 0)[0]
                        stack.extend(neighbors)
                
                components.append(component)
        
        if len(components) > 1:
            validation['warnings'].append(f"Network has {len(components)} disconnected components")
            
            # Check component sizes
            component_sizes = [len(comp) for comp in components]
            largest_component = max(component_sizes)
            
            if largest_component < n_states * 0.8:
                validation['warnings'].append("Largest component contains < 80% of states")
        
        # Check connectivity density
        n_edges = torch.sum(adjacency_matrix > 0).item() / 2  # Undirected graph
        max_edges = n_states * (n_states - 1) / 2
        density = n_edges / max_edges
        
        if density < 0.01:
            validation['warnings'].append("Very sparse network (< 1% edge density)")
        elif density > 0.5:
            validation['warnings'].append("Very dense network (> 50% edge density)")
        
        return validation


# =============================================================================
# Export all functions and classes
# =============================================================================

__all__ = [
    # Main validation functions (the missing ones)
    'check_installation',
    'check_gpu_availability',
    'validate_environment',
    'check_dependencies',
    
    # Validator classes
    'StructureValidator',
    'ResultValidator',
    
    # Individual validation methods
    'validate_protein_structure',
    'validate_ligand_molecule', 
    'validate_binding_sites',
    'validate_kinetic_results',
    'validate_simulation_results',
    'validate_network_connectivity'
]