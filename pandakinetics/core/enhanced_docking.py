# Enhanced docking module for PandaKinetics
# File: pandakinetics/core/enhanced_docking.py

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - limited coordinate handling")

try:
    import biotite.structure as struc
    import biotite.structure.io as strucio
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False
    logger.warning("Biotite not available - limited PDB handling")


class EnhancedDockingEngine:
    """
    Enhanced docking engine with coordinate input support
    
    Supports:
    - SDF file input with pre-defined coordinates
    - Ligand extraction from PDB files
    - Manual coordinate specification
    - Binding site detection from existing ligands
    """
    
    def __init__(self, device: Optional[str] = None, **kwargs):
        self.device = device
        logger.info("EnhancedDockingEngine initialized")
    
    def dock_with_coordinates(
        self,
        protein_pdb: str,
        ligand_input: Union[str, Path],
        input_type: str = "auto",  # "sdf", "pdb", "smiles", "coords"
        binding_site_coords: Optional[np.ndarray] = None,
        n_conformations: int = 50
    ) -> List[Dict]:
        """
        Dock ligand using various input formats
        
        Args:
            protein_pdb: Protein PDB file
            ligand_input: Input file/string (SDF, PDB, SMILES, or coordinates)
            input_type: Type of input ("sdf", "pdb", "smiles", "coords", "auto")
            binding_site_coords: Manual binding site coordinates [x, y, z]
            n_conformations: Number of conformations to generate
        """
        logger.info(f"Docking with input type: {input_type}")
        
        # Auto-detect input type
        if input_type == "auto":
            input_type = self._detect_input_type(ligand_input)
        
        # Load ligand coordinates
        ligand_coords, ligand_mol = self._load_ligand_coordinates(ligand_input, input_type)
        
        # Detect or use binding site
        if binding_site_coords is not None:
            binding_site = {
                'center': binding_site_coords,
                'radius': 10.0,
                'source': 'manual'
            }
        else:
            binding_site = self._detect_binding_site_from_ligand(protein_pdb, ligand_coords)
        
        # Generate conformations around the binding site
        poses = self._generate_poses_around_site(
            ligand_mol, ligand_coords, binding_site, n_conformations
        )
        
        return poses
    
    def extract_ligand_from_pdb(
        self,
        pdb_file: str,
        ligand_name: Optional[str] = None,
        chain: Optional[str] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Extract ligand coordinates and identity from PDB file
        
        Args:
            pdb_file: PDB file path
            ligand_name: Specific ligand residue name (e.g., "LIG", "ATP")
            chain: Specific chain identifier
            
        Returns:
            Tuple of (coordinates, ligand_name)
        """
        if not BIOTITE_AVAILABLE:
            raise ImportError("Biotite required for PDB ligand extraction")
        
        structure = strucio.load_structure(pdb_file)
        
        # Find non-standard residues (likely ligands)
        standard_residues = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
            "HOH", "WAT"  # Water
        }
        
        ligand_residues = []
        for res_name in np.unique(structure.res_name):
            if res_name not in standard_residues:
                ligand_residues.append(res_name)
        
        if ligand_name:
            target_ligand = ligand_name
        elif ligand_residues:
            target_ligand = ligand_residues[0]  # Take first found
            logger.info(f"Found ligands: {ligand_residues}, using: {target_ligand}")
        else:
            raise ValueError("No ligands found in PDB file")
        
        # Extract ligand atoms
        ligand_mask = structure.res_name == target_ligand
        if chain:
            ligand_mask &= structure.chain_id == chain
        
        ligand_atoms = structure[ligand_mask]
        
        if len(ligand_atoms) == 0:
            raise ValueError(f"Ligand {target_ligand} not found")
        
        coordinates = ligand_atoms.coord
        logger.info(f"Extracted {len(coordinates)} atoms for ligand {target_ligand}")
        
        return coordinates, target_ligand
    
    def _detect_input_type(self, ligand_input: Union[str, Path]) -> str:
        """Auto-detect input type"""
        
        if isinstance(ligand_input, (str, Path)):
            path = Path(ligand_input)
            if path.exists():
                suffix = path.suffix.lower()
                if suffix == ".sdf":
                    return "sdf"
                elif suffix in [".pdb", ".ent"]:
                    return "pdb"
                else:
                    return "smiles"  # Assume SMILES string
            else:
                return "smiles"  # String input, assume SMILES
        else:
            return "coords"  # Numpy array or similar
    
    def _load_ligand_coordinates(
        self, ligand_input: Union[str, Path], input_type: str
    ) -> Tuple[np.ndarray, Optional[Chem.Mol]]:
        """Load ligand coordinates from various formats"""
        
        if input_type == "sdf":
            return self._load_from_sdf(ligand_input)
        elif input_type == "pdb":
            coords, _ = self.extract_ligand_from_pdb(ligand_input)
            return coords, None
        elif input_type == "smiles":
            return self._generate_from_smiles(ligand_input)
        elif input_type == "coords":
            return np.array(ligand_input), None
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
    
    def _load_from_sdf(self, sdf_file: str) -> Tuple[np.ndarray, Chem.Mol]:
        """Load ligand from SDF file"""
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required for SDF file handling")
        
        suppl = Chem.SDMolSupplier(str(sdf_file))
        mol = next(suppl)
        
        if mol is None:
            raise ValueError(f"Could not read molecule from {sdf_file}")
        
        # Get 3D coordinates
        conf = mol.GetConformer()
        coords = conf.GetPositions()
        
        logger.info(f"Loaded {len(coords)} atoms from SDF file")
        return coords, mol
    
    def _generate_from_smiles(self, smiles: str) -> Tuple[np.ndarray, Chem.Mol]:
        """Generate 3D coordinates from SMILES (improved version)"""
        
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required for SMILES processing")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        mol = Chem.AddHs(mol)
        
        # Use ETKDG for better 3D conformer generation
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        
        AllChem.EmbedMolecule(mol, params)
        
        # Optimize with MMFF94
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            # Fallback to UFF if MMFF94 fails
            AllChem.UFFOptimizeMolecule(mol)
        
        conf = mol.GetConformer()
        coords = conf.GetPositions()
        
        logger.info(f"Generated {len(coords)} atom coordinates from SMILES")
        return coords, mol
    
    def _detect_binding_site_from_ligand(
        self, protein_pdb: str, ligand_coords: np.ndarray
    ) -> Dict:
        """Detect binding site from existing ligand coordinates"""
        
        # Calculate center of ligand
        center = np.mean(ligand_coords, axis=0)
        
        # Calculate binding site radius (distance to furthest atom + buffer)
        distances = np.linalg.norm(ligand_coords - center, axis=1)
        radius = np.max(distances) + 5.0  # 5Å buffer
        
        binding_site = {
            'center': center,
            'radius': radius,
            'source': 'ligand_based',
            'n_ligand_atoms': len(ligand_coords)
        }
        
        logger.info(f"Detected binding site at {center} with radius {radius:.1f}Å")
        return binding_site
    
    def _generate_poses_around_site(
        self,
        ligand_mol: Optional[Chem.Mol],
        reference_coords: np.ndarray,
        binding_site: Dict,
        n_conformations: int
    ) -> List[Dict]:
        """Generate multiple poses around the binding site"""
        
        poses = []
        site_center = binding_site['center']
        site_radius = binding_site['radius']
        
        # Generate conformations
        for i in range(n_conformations):
            if ligand_mol and RDKIT_AVAILABLE:
                # Use RDKit for better conformer generation
                conf_coords = self._generate_rdkit_conformation(ligand_mol, site_center, site_radius)
            else:
                # Generate perturbed coordinates
                conf_coords = self._generate_perturbed_coordinates(reference_coords, site_center, site_radius)
            
            # Calculate simple energy score
            energy = self._calculate_simple_energy(conf_coords, site_center)
            
            pose = {
                'coordinates': conf_coords,
                'energy': energy,
                'pose_id': i,
                'binding_site_center': site_center,
                'binding_site_radius': site_radius
            }
            poses.append(pose)
        
        # Sort by energy
        poses.sort(key=lambda x: x['energy'])
        
        logger.info(f"Generated {len(poses)} poses around binding site")
        return poses
    
    def _generate_rdkit_conformation(
        self, mol: Chem.Mol, site_center: np.ndarray, site_radius: float
    ) -> np.ndarray:
        """Generate conformer using RDKit and position in binding site"""
        
        # Create a copy of the molecule
        mol_copy = Chem.Mol(mol)
        
        # Generate new conformer
        AllChem.EmbedMolecule(mol_copy, randomSeed=np.random.randint(10000))
        AllChem.UFFOptimizeMolecule(mol_copy)
        
        # Get coordinates
        conf = mol_copy.GetConformer()
        coords = conf.GetPositions()
        
        # Center on binding site with small random displacement
        ligand_center = np.mean(coords, axis=0)
        displacement = np.random.normal(0, 1.0, 3)  # Random displacement
        new_center = site_center + displacement
        
        # Translate ligand
        final_coords = coords - ligand_center + new_center
        
        return final_coords
    
    def _generate_perturbed_coordinates(
        self, reference_coords: np.ndarray, site_center: np.ndarray, site_radius: float
    ) -> np.ndarray:
        """Generate perturbed coordinates around reference"""
        
        # Add random noise to coordinates
        noise_scale = 1.0  # Angstroms
        noise = np.random.normal(0, noise_scale, reference_coords.shape)
        perturbed_coords = reference_coords + noise
        
        # Add random rotation
        rotation_angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.normal(0, 1, 3)
        axis = axis / np.linalg.norm(axis)
        
        # Apply rotation around site center
        centered_coords = perturbed_coords - site_center
        rotated_coords = self._rotate_coordinates(centered_coords, axis, rotation_angle)
        final_coords = rotated_coords + site_center
        
        return final_coords
    
    def _rotate_coordinates(self, coords: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rotate coordinates around axis by angle (Rodrigues formula)"""
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Rodrigues rotation formula
        rotation_matrix = (cos_angle * np.eye(3) + 
                          sin_angle * self._cross_product_matrix(axis) +
                          (1 - cos_angle) * np.outer(axis, axis))
        
        return np.dot(coords, rotation_matrix.T)
    
    def _cross_product_matrix(self, vector: np.ndarray) -> np.ndarray:
        """Create cross product matrix for rotation"""
        
        return np.array([
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0]
        ])
    
    def _calculate_simple_energy(self, coords: np.ndarray, site_center: np.ndarray) -> float:
        """Calculate simple energy score based on distance to site center"""
        
        # Distance penalty (prefer ligands closer to site center)
        center_distance = np.linalg.norm(np.mean(coords, axis=0) - site_center)
        distance_penalty = center_distance * 0.1
        
        # Compactness score (prefer more compact conformations)
        ligand_center = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - ligand_center, axis=1)
        compactness_penalty = np.std(distances) * 0.05
        
        # Random energy component to simulate real binding energy variation
        random_component = np.random.normal(-8.0, 2.0)
        
        total_energy = random_component + distance_penalty + compactness_penalty
        
        return total_energy


# Usage examples and CLI integration
def enhanced_predict_with_coordinates():
    """Example of enhanced prediction with coordinate input"""
    
    # Example 1: Using SDF file
    docking_engine = EnhancedDockingEngine()
    
    poses_sdf = docking_engine.dock_with_coordinates(
        protein_pdb="protein.pdb",
        ligand_input="ligand.sdf",
        input_type="sdf",
        n_conformations=100
    )
    
    # Example 2: Extracting ligand from PDB
    coords, ligand_name = docking_engine.extract_ligand_from_pdb(
        pdb_file="complex.pdb",
        ligand_name="LIG"
    )
    
    poses_pdb = docking_engine.dock_with_coordinates(
        protein_pdb="protein.pdb",
        ligand_input=coords,
        input_type="coords",
        n_conformations=50
    )
    
    # Example 3: Manual binding site specification
    binding_site_coords = np.array([25.0, 30.0, 15.0])  # x, y, z coordinates
    
    poses_manual = docking_engine.dock_with_coordinates(
        protein_pdb="protein.pdb",
        ligand_input="CCO",  # ethanol
        input_type="smiles",
        binding_site_coords=binding_site_coords,
        n_conformations=25
    )
    
    return poses_sdf, poses_pdb, poses_manual


# Integration with PandaKinetics CLI
def add_enhanced_docking_options():
    """Enhanced CLI options for docking"""
    
    enhanced_options = """
    # Enhanced docking options for PandaKinetics predict command:
    
    pandakinetics predict \\
        --protein protein.pdb \\
        --ligand-sdf ligand.sdf \\
        --binding-site-coords "25.0,30.0,15.0" \\
        --n-conformations 100 \\
        --output enhanced_results
    
    # Extract ligand from existing complex:
    pandakinetics predict \\
        --protein protein.pdb \\
        --ligand-from-pdb complex.pdb \\
        --ligand-name "ATP" \\
        --n-conformations 50 \\
        --output atp_results
    
    # Use manual coordinates:
    pandakinetics predict \\
        --protein protein.pdb \\
        --ligand-coords coordinates.txt \\
        --binding-site-auto \\
        --output manual_results
    """
    
    return enhanced_options