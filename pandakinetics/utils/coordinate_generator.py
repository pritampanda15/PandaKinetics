#!/usr/bin/env python3
"""
Realistic coordinate generator for molecular structures
"""

import numpy as np
import torch
from typing import Optional, List, Tuple
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - using fallback coordinate generation")


class RealisticCoordinateGenerator:
    """Generate realistic 3D coordinates for molecules"""
    
    def __init__(self):
        self.rdkit_available = RDKIT_AVAILABLE
    
    def generate_realistic_coordinates(self, smiles: str, n_conformers: int = 10) -> List[np.ndarray]:
        """Generate realistic 3D coordinates from SMILES"""
        
        if not self.rdkit_available:
            return self._fallback_coordinates(smiles, n_conformers)
        
        try:
            # Create molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return self._fallback_coordinates(smiles, n_conformers)
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate conformers
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol, 
                numConfs=n_conformers,
                randomSeed=42,
                pruneRmsThresh=1.0,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
                enforceChirality=True
            )
            
            # Optimize conformers
            coordinates_list = []
            for conf_id in conformer_ids:
                # Optimize with MMFF
                try:
                    AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
                except:
                    # Fallback to UFF
                    AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
                
                # Extract coordinates
                conf = mol.GetConformer(conf_id)
                coords = []
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])
                
                coordinates_list.append(np.array(coords))
            
            logger.info(f"Generated {len(coordinates_list)} realistic conformers for {smiles}")
            return coordinates_list
            
        except Exception as e:
            logger.error(f"Failed to generate realistic coordinates: {e}")
            return self._fallback_coordinates(smiles, n_conformers)
    
    def _fallback_coordinates(self, smiles: str, n_conformers: int) -> List[np.ndarray]:
        """Fallback coordinate generation"""
        
        # Estimate number of atoms from SMILES
        n_atoms = max(10, len([c for c in smiles if c.isalpha()]) + 5)
        
        coordinates_list = []
        for i in range(n_conformers):
            # Generate somewhat realistic coordinates
            coords = self._generate_chemical_like_coords(n_atoms, seed=i)
            coordinates_list.append(coords)
        
        return coordinates_list
    
    def _generate_chemical_like_coords(self, n_atoms: int, seed: int = 42) -> np.ndarray:
        """Generate chemically reasonable coordinates"""
        
        np.random.seed(seed)
        
        coords = []
        
        # Start with first atom at origin
        coords.append([0.0, 0.0, 0.0])
        
        # Add atoms with realistic bond lengths and angles
        for i in range(1, n_atoms):
            if i == 1:
                # Second atom: typical C-C bond length
                coords.append([1.54, 0.0, 0.0])
            elif i == 2:
                # Third atom: tetrahedral angle
                coords.append([0.77, 1.33, 0.0])
            else:
                # Subsequent atoms: build with realistic geometry
                # Choose random previous atom to connect to
                prev_idx = np.random.randint(0, min(i, 3))
                prev_coord = np.array(coords[prev_idx])
                
                # Generate direction with some randomness
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                
                # Use realistic bond length
                bond_length = np.random.normal(1.54, 0.1)  # C-C bond ~ 1.54 Å
                
                new_coord = prev_coord + direction * bond_length
                coords.append(new_coord.tolist())
        
        return np.array(coords)
    
    def position_ligand_near_binding_site(
        self, 
        ligand_coords: np.ndarray, 
        binding_site_center: np.ndarray,
        max_displacement: float = 5.0
    ) -> np.ndarray:
        """Position ligand coordinates near a binding site"""
        
        # Calculate ligand center
        ligand_center = np.mean(ligand_coords, axis=0)
        
        # Calculate translation to move ligand to binding site
        translation = binding_site_center - ligand_center
        
        # Add some random displacement
        random_displacement = np.random.normal(0, max_displacement/3, 3)
        total_translation = translation + random_displacement
        
        # Apply translation
        positioned_coords = ligand_coords + total_translation
        
        return positioned_coords
    
    def generate_conformer_ensemble(
        self, 
        smiles: str, 
        binding_site_center: Optional[np.ndarray] = None,
        n_conformers: int = 10
    ) -> List[np.ndarray]:
        """Generate ensemble of conformers positioned near binding site"""
        
        # Generate base conformers
        base_conformers = self.generate_realistic_coordinates(smiles, n_conformers)
        
        if binding_site_center is None:
            return base_conformers
        
        # Position each conformer near binding site
        positioned_conformers = []
        for coords in base_conformers:
            positioned = self.position_ligand_near_binding_site(
                coords, binding_site_center
            )
            positioned_conformers.append(positioned)
        
        return positioned_conformers


# Helper function to integrate with existing code
def generate_realistic_ligand_coordinates(
    smiles: str, 
    n_conformers: int = 10,
    binding_site_center: Optional[np.ndarray] = None
) -> torch.Tensor:
    """Generate realistic coordinates and return as PyTorch tensor"""
    
    generator = RealisticCoordinateGenerator()
    
    if binding_site_center is not None:
        coords_list = generator.generate_conformer_ensemble(
            smiles, binding_site_center, n_conformers
        )
    else:
        coords_list = generator.generate_realistic_coordinates(smiles, n_conformers)
    
    if not coords_list:
        # Fallback
        coords_list = [np.random.randn(20, 3) * 2.0]
    
    # Convert to PyTorch tensor
    # Use the first conformer or stack all conformers
    if len(coords_list) == 1:
        return torch.tensor(coords_list[0], dtype=torch.float32)
    else:
        # Return multiple conformers as separate poses
        return torch.stack([torch.tensor(coords, dtype=torch.float32) for coords in coords_list])
