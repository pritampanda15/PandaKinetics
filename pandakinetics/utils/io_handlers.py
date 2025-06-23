# =============================================================================
# pandakinetics/utils/io_handlers.py
# =============================================================================

import numpy as np
from typing import Optional, Union, List, Dict
import biotite.structure as struc
import biotite.structure.io as strucio
from biotite.database import rcsb
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile
import os
from loguru import logger


class PDBHandler:
    """Handler for PDB structure files and operations"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"PDBHandler initialized with temp dir: {self.temp_dir}")
    
    def fetch_pdb(self, pdb_id: str) -> struc.AtomArray:
        """Fetch PDB structure from RCSB database"""
        
        try:
            # Download PDB file
            pdb_file = rcsb.fetch(pdb_id, "pdb", self.temp_dir)
            
            # Load structure
            structure = strucio.load_structure(pdb_file)
            
            # Take first model if NMR structure
            if isinstance(structure, struc.AtomArrayStack):
                structure = structure[0]
            
            logger.info(f"Fetched PDB {pdb_id}: {len(structure)} atoms")
            return structure
            
        except Exception as e:
            logger.error(f"Failed to fetch PDB {pdb_id}: {e}")
            raise
    
    def load_pdb(self, pdb_path: str) -> struc.AtomArray:
        """Load PDB structure from file"""
        
        try:
            structure = strucio.load_structure(pdb_path)
            
            # Take first model if multiple models
            if isinstance(structure, struc.AtomArrayStack):
                structure = structure[0]
            
            logger.info(f"Loaded PDB from {pdb_path}: {len(structure)} atoms")
            return structure
            
        except Exception as e:
            logger.error(f"Failed to load PDB from {pdb_path}: {e}")
            raise
    
    def clean_structure(self, structure: struc.AtomArray) -> struc.AtomArray:
        """Clean and prepare protein structure"""
        
        # Remove water molecules
        structure = structure[structure.res_name != "HOH"]
        
        # Remove other solvent molecules
        solvent_names = ["WAT", "TIP", "SOL", "CL", "NA", "K", "MG", "CA", "ZN"]
        for solvent in solvent_names:
            structure = structure[structure.res_name != solvent]
        
        # Keep only standard amino acids for protein
        standard_aa = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
        ]
        
        protein_mask = np.isin(structure.res_name, standard_aa)
        structure = structure[protein_mask]
        
        # Add missing hydrogen atoms if needed
        # This is simplified - in practice would use more sophisticated methods
        
        logger.info(f"Cleaned structure: {len(structure)} atoms")
        return structure
    
    def save_pdb(self, structure: struc.AtomArray, filename: str):
        """Save structure to PDB file"""
        
        try:
            strucio.save_structure(filename, structure)
            logger.info(f"Structure saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save structure to {filename}: {e}")
            raise
    
    def get_binding_site_residues(
        self, 
        structure: struc.AtomArray, 
        ligand_coords: np.ndarray, 
        cutoff: float = 5.0
    ) -> List[int]:
        """Get residues within cutoff distance of ligand"""
        
        binding_site_residues = []
        
        for res_id in np.unique(structure.res_id):
            residue = structure[structure.res_id == res_id]
            res_coords = residue.coord
            
            # Calculate minimum distance to ligand
            distances = np.linalg.norm(
                res_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :], 
                axis=2
            )
            min_distance = np.min(distances)
            
            if min_distance <= cutoff:
                binding_site_residues.append(res_id)
        
        return binding_site_residues


class MoleculeHandler:
    """Handler for molecular structures and chemical operations"""
    
    def __init__(self):
        logger.info("MoleculeHandler initialized")
    
    def smiles_to_mol(self, smiles: str) -> Chem.Mol:
        """Convert SMILES string to RDKit molecule"""
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            logger.info(f"Created molecule from SMILES: {smiles}")
            return mol
            
        except Exception as e:
            logger.error(f"Failed to create molecule from SMILES {smiles}: {e}")
            raise
    
    def generate_conformers(
        self, 
        mol: Chem.Mol, 
        n_conf: int = 10,
        energy_window: float = 10.0
    ) -> List[Chem.Mol]:
        """Generate diverse conformers for molecule"""
        
        try:
            # Generate conformers
            conf_ids = AllChem.EmbedMultipleConfs(
                mol, 
                numConfs=n_conf,
                randomSeed=42,
                pruneRmsThresh=1.0,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True
            )
            
            # Optimize conformers
            energies = []
            for conf_id in conf_ids:
                energy = AllChem.UFFOptimizeMolecule(mol, confId=conf_id)
                energies.append(energy)
            
            # Filter by energy window
            min_energy = min(energies)
            valid_conformers = []
            
            for i, (conf_id, energy) in enumerate(zip(conf_ids, energies)):
                if energy <= min_energy + energy_window:
                    # Create new molecule with single conformer
                    conf_mol = Chem.Mol(mol)
                    conf_mol.RemoveAllConformers()
                    conf_mol.AddConformer(mol.GetConformer(conf_id), assignId=True)
                    valid_conformers.append(conf_mol)
            
            logger.info(f"Generated {len(valid_conformers)} conformers")
            return valid_conformers
            
        except Exception as e:
            logger.error(f"Failed to generate conformers: {e}")
            raise
    
    def calculate_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate molecular descriptors"""
        
        from rdkit.Chem import Descriptors, Crippen, Lipinski
        
        descriptors = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'hbd': Lipinski.NumHDonors(mol),
            'hba': Lipinski.NumHAcceptors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'tpsa': Descriptors.TPSA(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol),
            'rings': Descriptors.RingCount(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol)
        }
        
        return descriptors
    
    def check_drug_likeness(self, mol: Chem.Mol) -> Dict[str, bool]:
        """Check drug-likeness rules"""
        
        descriptors = self.calculate_descriptors(mol)
        
        # Lipinski's Rule of Five
        lipinski = {
            'mw_ok': descriptors['molecular_weight'] <= 500,
            'logp_ok': descriptors['logp'] <= 5,
            'hbd_ok': descriptors['hbd'] <= 5,
            'hba_ok': descriptors['hba'] <= 10
        }
        
        # Veber rules
        veber = {
            'rotbonds_ok': descriptors['rotatable_bonds'] <= 10,
            'tpsa_ok': descriptors['tpsa'] <= 140
        }
        
        # Combined assessment
        drug_like = {
            'lipinski_compliant': all(lipinski.values()),
            'veber_compliant': all(veber.values()),
            'overall_drug_like': all(lipinski.values()) and all(veber.values())
        }
        
        return {**lipinski, **veber, **drug_like}