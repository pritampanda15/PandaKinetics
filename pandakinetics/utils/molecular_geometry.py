# Fixed coordinate generation for PandaKinetics
# File: pandakinetics/utils/molecular_geometry.py

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors, rdDistGeom
    from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule, MMFFOptimizeMolecule
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - using fallback geometry")


class MolecularGeometryGenerator:
    """
    Generate realistic molecular geometries for ligands
    
    Replaces random coordinate generation with proper 3D molecular structures
    """
    
    def __init__(self):
        self.bond_lengths = {
            ('C', 'C'): 1.54,   # sp3-sp3
            ('C', 'N'): 1.47,   # C-N single
            ('C', 'O'): 1.43,   # C-O single
            ('C', 'S'): 1.81,   # C-S single
            ('N', 'N'): 1.45,   # N-N single
            ('N', 'O'): 1.40,   # N-O single
            ('O', 'O'): 1.48,   # O-O single (peroxide)
            ('C', 'H'): 1.09,   # C-H
            ('N', 'H'): 1.01,   # N-H
            ('O', 'H'): 0.96,   # O-H
        }
        
        self.bond_angles = {
            'sp3': 109.47,  # tetrahedral
            'sp2': 120.0,   # trigonal planar
            'sp': 180.0,    # linear
        }
        
    def generate_realistic_coordinates(
        self, 
        smiles: str, 
        n_conformers: int = 10,
        energy_window: float = 10.0
    ) -> List[np.ndarray]:
        """
        Generate realistic 3D coordinates from SMILES
        
        Args:
            smiles: SMILES string
            n_conformers: Number of conformers to generate
            energy_window: Energy window for conformer selection (kcal/mol)
            
        Returns:
            List of coordinate arrays
        """
        if RDKIT_AVAILABLE:
            return self._generate_rdkit_conformers(smiles, n_conformers, energy_window)
        else:
            return self._generate_fallback_coordinates(smiles, n_conformers)
    
    def _generate_rdkit_conformers(
        self, smiles: str, n_conformers: int, energy_window: float
    ) -> List[np.ndarray]:
        """Generate conformers using RDKit's advanced methods"""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        mol = Chem.AddHs(mol)
        
        # Use ETKDG v3 for better conformer generation
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        params.maxAttempts = 1000
        params.numThreads = 0  # Use all available threads
        params.useRandomCoords = True
        params.boxSizeMult = 2.0
        
        # Generate initial conformers
        conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n_conformers * 2, params=params)
        
        if not conf_ids:
            logger.warning("Failed to generate conformers with ETKDG, using basic method")
            return self._generate_basic_rdkit_conformers(mol, n_conformers)
        
        # Optimize conformers with force field
        optimized_conformers = []
        energies = []
        
        for conf_id in conf_ids:
            try:
                # Try MMFF94 first (more accurate)
                if MMFFOptimizeMolecule(mol, confId=conf_id) == 0:
                    # Get MMFF energy
                    ff = AllChem.MMFFGetMoleculeForceField(mol, confId=conf_id)
                    if ff:
                        energy = ff.CalcEnergy()
                    else:
                        energy = 0.0
                else:
                    # Fallback to UFF
                    UFFOptimizeMolecule(mol, confId=conf_id)
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                    energy = ff.CalcEnergy() if ff else 0.0
                
                conf = mol.GetConformer(conf_id)
                coords = conf.GetPositions()
                
                optimized_conformers.append(coords)
                energies.append(energy)
                
            except Exception as e:
                logger.warning(f"Failed to optimize conformer {conf_id}: {e}")
                continue
        
        if not optimized_conformers:
            logger.warning("No conformers successfully optimized")
            return self._generate_basic_rdkit_conformers(mol, n_conformers)
        
        # Filter by energy window
        energies = np.array(energies)
        min_energy = np.min(energies)
        
        filtered_conformers = []
        for i, (coords, energy) in enumerate(zip(optimized_conformers, energies)):
            if energy <= min_energy + energy_window:
                filtered_conformers.append(coords)
        
        # Cluster by RMSD to get diverse conformers
        diverse_conformers = self._cluster_conformers(filtered_conformers, rmsd_threshold=1.0)
        
        # Return requested number of conformers
        return diverse_conformers[:n_conformers]
    
    def _generate_basic_rdkit_conformers(self, mol: Chem.Mol, n_conformers: int) -> List[np.ndarray]:
        """Basic conformer generation as fallback"""
        
        conformers = []
        
        for i in range(n_conformers):
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42 + i)
                UFFOptimizeMolecule(mol)
                
                conf = mol.GetConformer()
                coords = conf.GetPositions()
                conformers.append(coords)
                
            except Exception as e:
                logger.warning(f"Failed to generate conformer {i}: {e}")
                continue
        
        return conformers
    
    def _generate_fallback_coordinates(self, smiles: str, n_conformers: int) -> List[np.ndarray]:
        """Fallback coordinate generation without RDKit"""
        
        # Estimate number of atoms from SMILES
        n_atoms = self._estimate_atom_count(smiles)
        
        conformers = []
        
        for i in range(n_conformers):
            # Generate more realistic coordinates using chemical knowledge
            coords = self._generate_chemical_coordinates(n_atoms, i)
            conformers.append(coords)
        
        return conformers
    
    def _estimate_atom_count(self, smiles: str) -> int:
        """Estimate atom count from SMILES string"""
        
        # Count explicit atoms (letters)
        atom_count = 0
        i = 0
        while i < len(smiles):
            char = smiles[i]
            if char.isupper():  # Start of atom symbol
                atom_count += 1
                # Skip lowercase letters (part of atom symbol)
                i += 1
                while i < len(smiles) and smiles[i].islower():
                    i += 1
            else:
                i += 1
        
        # Add implicit hydrogens (rough estimate)
        # This is very approximate - real calculation needs proper parsing
        estimated_hydrogens = max(1, atom_count // 2)
        
        return atom_count + estimated_hydrogens
    
    def _generate_chemical_coordinates(self, n_atoms: int, seed: int) -> np.ndarray:
        """Generate chemically reasonable coordinates"""
        
        np.random.seed(42 + seed)
        
        if n_atoms <= 1:
            return np.array([[0.0, 0.0, 0.0]])
        
        coords = np.zeros((n_atoms, 3))
        
        # Place first atom at origin
        coords[0] = [0.0, 0.0, 0.0]
        
        if n_atoms == 1:
            return coords
        
        # Place second atom at typical bond distance
        bond_length = 1.5  # Typical C-C bond length
        coords[1] = [bond_length, 0.0, 0.0]
        
        # Build molecule using reasonable geometry
        for i in range(2, n_atoms):
            # Connect to previous atom with realistic bond length and angle
            
            # Choose bond length based on typical values
            bond_length = np.random.normal(1.5, 0.1)  # Around 1.5 Å
            bond_length = max(bond_length, 1.0)  # Minimum reasonable bond length
            
            # Choose bond angle (tetrahedral is most common)
            if i == 2:
                # Third atom: use tetrahedral angle
                angle = np.radians(109.47 + np.random.normal(0, 10))
            else:
                # Subsequent atoms: varied angles
                angle = np.radians(np.random.uniform(100, 130))
            
            # Place atom using spherical coordinates
            if i >= 3:
                # Use torsion angle for 4th atom onwards
                torsion = np.random.uniform(0, 2 * np.pi)
            else:
                torsion = 0.0
            
            # Calculate position relative to previous two atoms
            prev_atom = coords[i-1]
            if i == 2:
                # For third atom, use vector from first to second atom
                ref_vector = coords[1] - coords[0]
            else:
                ref_vector = coords[i-1] - coords[i-2]
            
            # Normalize reference vector
            ref_vector = ref_vector / np.linalg.norm(ref_vector)
            
            # Create perpendicular vector
            if abs(ref_vector[2]) < 0.9:
                perp_vector = np.array([0, 0, 1])
            else:
                perp_vector = np.array([1, 0, 0])
            
            # Make it truly perpendicular
            perp_vector = perp_vector - np.dot(perp_vector, ref_vector) * ref_vector
            perp_vector = perp_vector / np.linalg.norm(perp_vector)
            
            # Create another perpendicular vector
            perp_vector2 = np.cross(ref_vector, perp_vector)
            
            # Calculate new atom position
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            direction = (-cos_angle * ref_vector + 
                        sin_angle * (np.cos(torsion) * perp_vector + 
                                   np.sin(torsion) * perp_vector2))
            
            coords[i] = prev_atom + bond_length * direction
        
        return coords
    
    def _cluster_conformers(self, conformers: List[np.ndarray], rmsd_threshold: float = 1.0) -> List[np.ndarray]:
        """Cluster conformers by RMSD to get diverse set"""
        
        if not conformers:
            return []
        
        if len(conformers) <= 1:
            return conformers
        
        # Simple clustering by RMSD
        clusters = [[conformers[0]]]
        
        for conf in conformers[1:]:
            # Find which cluster this conformer belongs to
            assigned = False
            
            for cluster in clusters:
                # Calculate RMSD to cluster representative (first member)
                rmsd = self._calculate_rmsd(conf, cluster[0])
                
                if rmsd < rmsd_threshold:
                    cluster.append(conf)
                    assigned = True
                    break
            
            if not assigned:
                # Create new cluster
                clusters.append([conf])
        
        # Return one representative from each cluster
        diverse_conformers = [cluster[0] for cluster in clusters]
        
        return diverse_conformers
    
    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate RMSD between two coordinate sets"""
        
        if coords1.shape != coords2.shape:
            return float('inf')
        
        # Center coordinates
        center1 = np.mean(coords1, axis=0)
        center2 = np.mean(coords2, axis=0)
        
        centered1 = coords1 - center1
        centered2 = coords2 - center2
        
        # Calculate RMSD
        diff = centered1 - centered2
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        
        return rmsd
    
    def create_realistic_pdb(
        self, coordinates: np.ndarray, smiles: str, output_file: str, 
        ligand_name: str = "LIG", energy: float = 0.0
    ):
        """Create a realistic PDB file with proper atom types"""
        
        if RDKIT_AVAILABLE:
            self._create_rdkit_pdb(coordinates, smiles, output_file, ligand_name, energy)
        else:
            self._create_simple_pdb(coordinates, output_file, ligand_name, energy)
    
    def _create_rdkit_pdb(
        self, coordinates: np.ndarray, smiles: str, output_file: str,
        ligand_name: str, energy: float
    ):
        """Create PDB with proper atom types using RDKit"""
        
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # Create conformer with our coordinates
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, coord in enumerate(coordinates):
            conf.SetAtomPosition(i, coord)
        
        mol.AddConformer(conf, assignId=True)
        
        # Generate PDB content
        pdb_lines = [
            "HEADER    PANDAKINETICS LIGAND STRUCTURE",
            f"TITLE     LIGAND {ligand_name} FROM SMILES: {smiles}",
            f"REMARK   BINDING ENERGY: {energy:.3f} KCAL/MOL",
            f"REMARK   GENERATED BY PANDAKINETICS MOLECULAR GEOMETRY",
            ""
        ]
        
        # Add atoms with correct types
        for i, atom in enumerate(mol.GetAtoms()):
            if i < len(coordinates):
                coord = coordinates[i]
                element = atom.GetSymbol()
                
                pdb_line = (
                    f"HETATM{i+1:5d} {element}{i+1:<3} {ligand_name} A   1    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00{abs(energy):6.2f}           {element:>2}"
                )
                pdb_lines.append(pdb_line)
        
        pdb_lines.append("END")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(pdb_lines) + '\n')
    
    def _create_simple_pdb(
        self, coordinates: np.ndarray, output_file: str, ligand_name: str, energy: float
    ):
        """Create simple PDB without RDKit"""
        
        pdb_lines = [
            "HEADER    PANDAKINETICS LIGAND STRUCTURE",
            f"TITLE     LIGAND {ligand_name}",
            f"REMARK   BINDING ENERGY: {energy:.3f} KCAL/MOL",
            ""
        ]
        
        # Use carbon as default atom type
        for i, coord in enumerate(coordinates):
            element = "C"  # Default to carbon
            pdb_line = (
                f"HETATM{i+1:5d} {element}{i+1:<3} {ligand_name} A   1    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00{abs(energy):6.2f}           {element:>2}"
            )
            pdb_lines.append(pdb_line)
        
        pdb_lines.append("END")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(pdb_lines) + '\n')


# Example usage function
def fix_existing_coordinates():
    """Example of how to fix existing coordinate generation"""
    
    generator = MolecularGeometryGenerator()
    
    # Generate realistic coordinates for common drug molecules
    molecules = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine
        "CCO",                             # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O"        # Aspirin
    ]
    
    for i, smiles in enumerate(molecules):
        try:
            conformers = generator.generate_realistic_coordinates(smiles, n_conformers=5)
            
            for j, coords in enumerate(conformers):
                output_file = f"realistic_ligand_{i}_{j}.pdb"
                generator.create_realistic_pdb(coords, smiles, output_file, f"MOL{i}", energy=-8.0 + j)
                print(f"Generated: {output_file}")
                
        except Exception as e:
            print(f"Failed to generate coordinates for {smiles}: {e}")


if __name__ == "__main__":
    fix_existing_coordinates()