#!/usr/bin/env python3
"""
Enhanced Visualization Module for PandaKinetics
File: pandakinetics/visualization/structure_export.py

Integration of transition state export with protein-ligand complexes
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import logging

# PandaKinetics imports
from pandakinetics.core.networks import TransitionNetwork
from pandakinetics.utils.io_handlers import PDBHandler, MoleculeHandler
from pandakinetics.simulation.results import SimulationResults

logger = logging.getLogger(__name__)

@dataclass
class TransitionStateExportConfig:
    """Configuration for transition state export"""
    include_protein: bool = True
    include_ligand: bool = True
    include_metadata: bool = True
    generate_pymol_script: bool = True
    generate_vmd_script: bool = False
    export_interactions: bool = True
    export_energies: bool = True
    output_format: str = "pdb"  # pdb, sdf, mol2
    coordinate_precision: int = 3
    # New positioning options
    position_ligand_near_protein: bool = True
    binding_site_detection: str = "auto"  # "auto", "center", "cavity"
    ligand_protein_distance: float = 5.0  # Target distance in Angstroms

class ProteinLigandComplexExporter:
    """
    Enhanced exporter for protein-ligand transition state complexes
    """
    
    def __init__(self, config: Optional[TransitionStateExportConfig] = None):
        self.config = config or TransitionStateExportConfig()
        self.pdb_handler = PDBHandler()
        self.mol_handler = MoleculeHandler()
        
    def export_transition_complexes(
        self,
        simulation_results: SimulationResults,
        transition_network: TransitionNetwork,
        protein_pdb_path: str,
        ligand_smiles: str,
        output_dir: Union[str, Path] = "transition_complexes"
    ) -> Dict[str, List[str]]:
        """
        Export all transition states as full protein-ligand complexes
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            transition_network: Network of transition states
            protein_pdb_path: Path to original protein PDB file
            ligand_smiles: SMILES string of ligand
            output_dir: Output directory for complexes
            
        Returns:
            Dictionary with lists of generated files
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting transition complexes to {output_path}")
        
        # Load protein structure
        protein_structure = self._load_protein_structure(protein_pdb_path)
        
        # Find protein binding site for ligand positioning
        binding_site_center = self._find_binding_site(protein_structure)
        logger.info(f"Binding site center: {binding_site_center}")
        
        # Export individual transition states
        exported_files = self._export_individual_states(
            transition_network, 
            protein_structure, 
            ligand_smiles, 
            output_path,
            binding_site_center
        )
        
        # Generate analysis files
        analysis_files = self._generate_analysis_files(
            simulation_results,
            transition_network,
            ligand_smiles,
            output_path
        )
        
        # Generate visualization scripts
        viz_files = self._generate_visualization_scripts(
            exported_files,
            output_path
        )
        
        return {
            "pdb_complexes": [f["pdb_file"] for f in exported_files],
            "metadata_files": [f["metadata_file"] for f in exported_files],
            "analysis_files": analysis_files,
            "visualization_scripts": viz_files
        }
    
    def _load_protein_structure(self, protein_pdb_path: str) -> Dict:
        """Load and parse protein structure"""
        
        with open(protein_pdb_path, 'r') as f:
            pdb_content = f.read()
        
        # Parse PDB to extract coordinates, residues, etc.
        protein_data = {
            "atoms": [],
            "residues": [],
            "coordinates": [],
            "original_content": pdb_content
        }
        
        for line in pdb_content.split('\n'):
            if line.startswith(('ATOM', 'HETATM')):
                atom_data = self._parse_pdb_line(line)
                if atom_data:  # Only add valid atom data
                    protein_data["atoms"].append(atom_data)
                    protein_data["coordinates"].append([
                        atom_data["x"], atom_data["y"], atom_data["z"]
                    ])
        
        protein_data["coordinates"] = np.array(protein_data["coordinates"])
        
        return protein_data
    
    def _parse_pdb_line(self, line: str) -> Optional[Dict]:
        """Parse a PDB ATOM/HETATM line with error handling"""
        
        try:
            return {
                "record": line[:6].strip(),
                "serial": int(line[6:11].strip()) if line[6:11].strip() else 1,
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21:22].strip() if len(line) > 21 else "A",
                "resseq": int(line[22:26].strip()) if line[22:26].strip() else 1,
                "x": float(line[30:38].strip()),
                "y": float(line[38:46].strip()),
                "z": float(line[46:54].strip()),
                "occupancy": float(line[54:60].strip()) if line[54:60].strip() else 1.0,
                "bfactor": float(line[60:66].strip()) if line[60:66].strip() else 0.0,
                "element": line[76:78].strip() if len(line) > 76 else "",
                "original_line": line
            }
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse PDB line: {line[:20]}... Error: {e}")
            return None
    
    def _find_binding_site(self, protein_structure: Dict) -> np.ndarray:
        """Find the protein binding site for ligand positioning"""
        
        protein_coords = protein_structure["coordinates"]
        
        if len(protein_coords) == 0:
            logger.warning("No protein coordinates found, using origin")
            return np.array([0.0, 0.0, 0.0])
        
        # Debug protein coordinate range
        min_coords = np.min(protein_coords, axis=0)
        max_coords = np.max(protein_coords, axis=0)
        center_coords = np.mean(protein_coords, axis=0)
        
        logger.info(f"Protein coordinate range: min={min_coords}, max={max_coords}, center={center_coords}")
        
        if self.config.binding_site_detection == "center":
            # Use geometric center of protein
            binding_site = np.mean(protein_coords, axis=0)
        elif self.config.binding_site_detection == "cavity":
            # Find largest cavity (simplified approach)
            binding_site = self._find_largest_cavity(protein_structure)
        else:  # "auto" - use heuristic approach
            binding_site = self._auto_detect_binding_site(protein_structure)
        
        logger.info(f"Selected binding site: {binding_site}")
        return binding_site
    
    def _auto_detect_binding_site(self, protein_structure: Dict) -> np.ndarray:
        """Auto-detect binding site using heuristic methods"""
        
        protein_coords = protein_structure["coordinates"]
        
        # Method 1: Find region with lowest density (potential cavity)
        # Divide space into grid and find least dense region
        min_coords = np.min(protein_coords, axis=0)
        max_coords = np.max(protein_coords, axis=0)
        
        # Create a coarse grid
        grid_size = 20
        x_grid = np.linspace(min_coords[0], max_coords[0], grid_size)
        y_grid = np.linspace(min_coords[1], max_coords[1], grid_size)
        z_grid = np.linspace(min_coords[2], max_coords[2], grid_size)
        
        best_cavity_score = float('inf')
        best_cavity_center = np.mean(protein_coords, axis=0)
        
        for i, x in enumerate(x_grid[1:-1], 1):  # Skip edges
            for j, y in enumerate(y_grid[1:-1], 1):
                for k, z in enumerate(z_grid[1:-1], 1):
                    test_point = np.array([x, y, z])
                    
                    # Count nearby atoms
                    distances = np.linalg.norm(protein_coords - test_point, axis=1)
                    nearby_atoms = np.sum(distances < 8.0)  # Within 8Å
                    very_close = np.sum(distances < 3.0)    # Within 3Å
                    
                    # Good binding site: some nearby atoms but not too crowded
                    if very_close == 0 and 5 <= nearby_atoms <= 30:
                        cavity_score = abs(nearby_atoms - 15)  # Optimal around 15 atoms
                        if cavity_score < best_cavity_score:
                            best_cavity_score = cavity_score
                            best_cavity_center = test_point
        
        logger.info(f"Auto-detected binding site with cavity score: {best_cavity_score}")
        return best_cavity_center
    
    def _find_largest_cavity(self, protein_structure: Dict) -> np.ndarray:
        """Find the largest cavity in the protein (simplified)"""
        
        # For now, return protein center
        # This could be enhanced with proper cavity detection algorithms
        protein_coords = protein_structure["coordinates"]
        return np.mean(protein_coords, axis=0)
    
    def _calculate_protein_center(self, protein_structure: Dict) -> np.ndarray:
        """Calculate the geometric center of the protein"""
        
        return np.mean(protein_structure["coordinates"], axis=0)
    
    def _position_ligand_near_binding_site(
        self, 
        ligand_coords: np.ndarray, 
        binding_site_center: np.ndarray,
        target_distance: float = None
    ) -> np.ndarray:
        """Position ligand coordinates near the binding site"""
        
        if target_distance is None:
            target_distance = self.config.ligand_protein_distance
        
        # Debug original ligand coordinates
        ligand_center = np.mean(ligand_coords, axis=0)
        ligand_min = np.min(ligand_coords, axis=0)
        ligand_max = np.max(ligand_coords, axis=0)
        
        logger.info(f"Original ligand center: {ligand_center}")
        logger.info(f"Original ligand range: min={ligand_min}, max={ligand_max}")
        logger.info(f"Binding site center: {binding_site_center}")
        
        # Calculate translation vector to move ligand to binding site
        translation_vector = binding_site_center - ligand_center
        logger.info(f"Translation vector: {translation_vector}")
        logger.info(f"Translation distance: {np.linalg.norm(translation_vector):.2f} Å")
        
        # Apply translation
        positioned_coords = ligand_coords + translation_vector
        
        # Optionally add small random displacement for diversity (but keep it small)
        random_displacement = np.random.randn(3) * 0.2  # Reduced from 0.5
        positioned_coords += random_displacement
        
        # Verify final position
        final_center = np.mean(positioned_coords, axis=0)
        distance_to_binding_site = np.linalg.norm(final_center - binding_site_center)
        
        logger.info(f"Final ligand center: {final_center}")
        logger.info(f"Distance to binding site: {distance_to_binding_site:.2f} Å")
        
        return positioned_coords
    
    def _export_individual_states(
        self,
        transition_network: TransitionNetwork,
        protein_structure: Dict,
        ligand_smiles: str,
        output_path: Path,
        binding_site_center: np.ndarray
    ) -> List[Dict]:
        """Export each transition state as a separate PDB complex"""
        
        exported_files = []
        n_states = len(transition_network.positions)
        
        logger.info(f"Exporting {n_states} transition states")
        
        for state_id in range(n_states):
            state_files = self._export_single_state(
                state_id,
                transition_network,
                protein_structure,
                ligand_smiles,
                output_path,
                binding_site_center
            )
            exported_files.append(state_files)
            
        return exported_files
    
    def _export_single_state(
        self,
        state_id: int,
        transition_network: TransitionNetwork,
        protein_structure: Dict,
        ligand_smiles: str,
        output_path: Path,
        binding_site_center: np.ndarray
    ) -> Dict:
        """Export a single transition state"""
        
        # Get state data
        ligand_coords = transition_network.positions[state_id].cpu().numpy()
        state_energy = transition_network.energies[state_id].item()
        
        # Position ligand near binding site
        if self.config.position_ligand_near_protein:
            positioned_ligand_coords = self._position_ligand_near_binding_site(
                ligand_coords, 
                binding_site_center
            )
        else:
            positioned_ligand_coords = ligand_coords
        
        # Generate filenames
        pdb_file = output_path / f"complex_state_{state_id:03d}.pdb"
        metadata_file = output_path / f"complex_state_{state_id:03d}_metadata.json"
        
        # Create full complex PDB
        complex_pdb = self._create_complex_pdb(
            protein_structure,
            positioned_ligand_coords,
            ligand_smiles,
            state_energy,
            state_id
        )
        
        # Write PDB file
        with open(pdb_file, 'w') as f:
            f.write(complex_pdb)
        
        # Create metadata
        metadata = self._create_state_metadata(
            state_id,
            state_energy,
            ligand_smiles,
            positioned_ligand_coords,
            transition_network
        )
        
        # Write metadata file
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Exported state {state_id}: {pdb_file}")
        
        return {
            "state_id": state_id,
            "pdb_file": str(pdb_file),
            "metadata_file": str(metadata_file),
            "energy": state_energy
        }
    
    def _create_complex_pdb(
        self,
        protein_structure: Dict,
        ligand_coords: np.ndarray,
        ligand_smiles: str,
        state_energy: float,
        state_id: int
    ) -> str:
        """Create a complete protein-ligand complex PDB with proper formatting"""
        
        lines = [
            "HEADER    PROTEIN-LIGAND COMPLEX                    " + "23-JUN-25   PKIN",
            f"TITLE     TRANSITION STATE {state_id} COMPLEX",
            f"REMARK   1 LIGAND SMILES: {ligand_smiles}",
            f"REMARK   2 BINDING ENERGY: {state_energy:.3f} KCAL/MOL",
            f"REMARK   3 STATE ID: {state_id}",
            f"REMARK   4 GENERATED BY PANDAKINETICS ENHANCED VISUALIZER",
            f"REMARK   5 LIGAND POSITIONED NEAR BINDING SITE: {self.config.position_ligand_near_protein}",
            f"REMARK   6 BINDING SITE DETECTION: {self.config.binding_site_detection}",
            "REMARK   7",
        ]
        
        # Add interaction analysis if ligand is properly positioned
        if self.config.position_ligand_near_protein:
            interactions = self._analyze_interactions(protein_structure, ligand_coords)
            lines.append("REMARK   8 PROTEIN-LIGAND INTERACTIONS:")
            for i, interaction in enumerate(interactions[:10]):  # Top 10 interactions
                lines.append(
                    f"REMARK   9 {interaction['protein_residue']} - "
                    f"LIG{interaction['ligand_atom']:>3} "
                    f"({interaction['distance']:.2f} A, {interaction['type']})"
                )
            lines.append("REMARK  10")
        
        # Add protein atoms with proper formatting
        protein_atom_count = 0
        last_residue_info = None
        
        for atom in protein_structure["atoms"]:
            # Modify B-factor to reflect interaction strength
            bfactor = self._calculate_interaction_bfactor(atom, ligand_coords)
            
            # Create properly formatted ATOM line
            atom_line = f"ATOM  {atom['serial']:5d} {atom['name']:>4} {atom['resname']:>3} {atom['chain']}{atom['resseq']:>4}    {atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}{atom['occupancy']:6.2f}{bfactor:6.2f}           {atom['element']:>2}"
            lines.append(atom_line)
            protein_atom_count += 1
            last_residue_info = (atom['resname'], atom['chain'], atom['resseq'])
        
        # Add proper TER record
        if last_residue_info:
            ter_line = f"TER   {protein_atom_count + 1:5d}      {last_residue_info[0]} {last_residue_info[1]}{last_residue_info[2]:>4}"
            lines.append(ter_line)
        
        # Generate ligand with proper atom names and connectivity
        ligand_data = self._generate_proper_ligand_structure(ligand_smiles, ligand_coords, state_energy)
        
        # Add ligand atoms with proper formatting
        ligand_start_serial = protein_atom_count + 2
        
        for i, atom_info in enumerate(ligand_data['atoms']):
            serial = ligand_start_serial + i
            atom_name = atom_info['name']
            element = atom_info['element']
            coord = atom_info['coord']
            
            hetatm_line = f"HETATM{serial:5d} {atom_name:>4} LIG L   1    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00{abs(state_energy):6.2f}           {element:>2}"
            lines.append(hetatm_line)
        
        # Add CONECT records for ligand bonds
        lines.extend(self._generate_conect_records(ligand_data['bonds'], ligand_start_serial))
        
        lines.append("END")
        return "\n".join(lines) + "\n"

    def _generate_proper_ligand_structure(self, smiles: str, coords: np.ndarray, energy: float) -> Dict:
        """Generate proper ligand structure with chemical atom names and bonds"""
        
        # Debug the coordinates being used
        logger.info(f"Generating ligand structure with {len(coords)} atoms")
        logger.info(f"Ligand coordinate range: min={np.min(coords, axis=0)}, max={np.max(coords, axis=0)}")
        logger.info(f"Ligand center: {np.mean(coords, axis=0)}")
        
        try:
            from rdkit import Chem
            
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning("Failed to create molecule from SMILES, using fallback")
                return self._fallback_ligand_structure(coords)
            
            mol = Chem.AddHs(mol)
            
            # Generate proper atom names based on chemical structure
            atoms = []
            element_counts = {}
            
            for i, atom in enumerate(mol.GetAtoms()):
                element = atom.GetSymbol()
                
                # Count elements for proper naming
                if element not in element_counts:
                    element_counts[element] = 0
                element_counts[element] += 1
                
                # Create chemically meaningful atom name
                if element == 'C':
                    atom_name = f"C{element_counts[element]}"
                elif element == 'N':
                    atom_name = f"N{element_counts[element]}"
                elif element == 'O':
                    atom_name = f"O{element_counts[element]}"
                elif element == 'S':
                    atom_name = f"S{element_counts[element]}"
                elif element == 'H':
                    atom_name = f"H{element_counts[element]}"
                else:
                    atom_name = f"{element}{element_counts[element]}"
                
                # CRITICAL: Use provided coordinates (these should be the positioned ones)
                if i < len(coords):
                    coord = coords[i].copy()  # Make sure we use the positioned coordinates
                else:
                    # Generate reasonable coordinates relative to existing ones
                    if len(coords) > 0:
                        # Generate near existing coordinates
                        base_coord = coords[0] if len(coords) > 0 else np.array([0., 0., 0.])
                        coord = base_coord + np.random.randn(3) * 2.0
                    else:
                        coord = np.random.randn(3) * 2.0
                
                atoms.append({
                    'name': atom_name,
                    'element': element,
                    'coord': coord,
                    'atom_idx': i
                })
            
            # Debug final atom coordinates
            if atoms:
                final_coords = np.array([atom['coord'] for atom in atoms])
                logger.info(f"Final ligand atom coordinates center: {np.mean(final_coords, axis=0)}")
            
            # Generate bond connectivity
            bonds = []
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                bonds.append((begin_idx, end_idx))
            
            return {
                'atoms': atoms,
                'bonds': bonds,
                'mol': mol
            }
            
        except ImportError:
            logger.warning("RDKit not available, using fallback ligand structure")
            return self._fallback_ligand_structure(coords)
        except Exception as e:
            logger.warning(f"Failed to generate proper ligand structure: {e}")
            return self._fallback_ligand_structure(coords)

    def _fallback_ligand_structure(self, coords: np.ndarray) -> Dict:
        """Fallback ligand structure when RDKit is not available"""
        
        atoms = []
        for i, coord in enumerate(coords):
            # Simple element assignment based on position
            if i < 10:
                element = 'C'
                atom_name = f"C{i+1}"
            elif i < 15:
                element = 'N'
                atom_name = f"N{i-9}"
            elif i < 18:
                element = 'O'
                atom_name = f"O{i-14}"
            else:
                element = 'H'
                atom_name = f"H{i-17}"
            
            atoms.append({
                'name': atom_name,
                'element': element,
                'coord': coord,
                'atom_idx': i
            })
        
        # Generate simple linear connectivity
        bonds = []
        for i in range(len(coords) - 1):
            bonds.append((i, i + 1))
        
        return {
            'atoms': atoms,
            'bonds': bonds
        }

    def _generate_conect_records(self, bonds: List[Tuple[int, int]], start_serial: int) -> List[str]:
        """Generate CONECT records for ligand bonds"""
        
        conect_lines = []
        
        # Group bonds by atom for proper CONECT format
        atom_connections = {}
        for bond in bonds:
            atom1, atom2 = bond
            serial1 = start_serial + atom1
            serial2 = start_serial + atom2
            
            if serial1 not in atom_connections:
                atom_connections[serial1] = []
            if serial2 not in atom_connections:
                atom_connections[serial2] = []
            
            atom_connections[serial1].append(serial2)
            atom_connections[serial2].append(serial1)
        
        # Generate CONECT lines
        for atom_serial in sorted(atom_connections.keys()):
            connected_atoms = sorted(atom_connections[atom_serial])
            
            # Format CONECT line (max 4 connections per line)
            for i in range(0, len(connected_atoms), 4):
                connections = connected_atoms[i:i+4]
                conect_line = f"CONECT{atom_serial:5d}"
                for conn in connections:
                    conect_line += f"{conn:5d}"
                conect_lines.append(conect_line)
        
        return conect_lines
    
    def _analyze_interactions(
        self, 
        protein_structure: Dict, 
        ligand_coords: np.ndarray
    ) -> List[Dict]:
        """Analyze protein-ligand interactions"""
        
        interactions = []
        protein_coords = protein_structure["coordinates"]
        
        # Calculate distances between all protein and ligand atoms
        for i, lig_coord in enumerate(ligand_coords):
            distances = np.linalg.norm(protein_coords - lig_coord, axis=1)
            
            # Find close contacts (< 4.0 Å)
            close_contacts = np.where(distances < 4.0)[0]
            
            for contact_idx in close_contacts:
                protein_atom = protein_structure["atoms"][contact_idx]
                distance = distances[contact_idx]
                
                # Determine interaction type
                interaction_type = self._determine_interaction_type(
                    protein_atom, distance
                )
                
                interactions.append({
                    "protein_residue": f"{protein_atom['resname']}{protein_atom['resseq']}",
                    "protein_atom": protein_atom["name"],
                    "ligand_atom": i,
                    "distance": distance,
                    "type": interaction_type
                })
        
        # Sort by distance
        interactions.sort(key=lambda x: x["distance"])
        
        return interactions
    
    def _determine_interaction_type(self, protein_atom: Dict, distance: float) -> str:
        """Determine the type of interaction based on atoms and distance"""
        
        if distance < 2.5:
            return "strong"
        elif distance < 3.5:
            if protein_atom["name"] in ["N", "O", "S"]:
                return "hydrogen_bond"
            else:
                return "van_der_waals"
        else:
            return "weak"
    
    def _calculate_interaction_bfactor(
        self, 
        protein_atom: Dict, 
        ligand_coords: np.ndarray
    ) -> float:
        """Calculate B-factor based on ligand interaction strength"""
        
        atom_coord = np.array([protein_atom["x"], protein_atom["y"], protein_atom["z"]])
        min_distance = np.min(np.linalg.norm(ligand_coords - atom_coord, axis=1))
        
        # Scale B-factor inversely with distance (closer = higher B-factor)
        if min_distance < 3.0:
            return min(99.99, 50.0 / min_distance)
        else:
            return protein_atom["bfactor"]
    
    def _generate_ligand_atom_names(self, n_atoms: int) -> List[str]:
        """Generate reasonable atom names for ligand"""
        
        names = []
        for i in range(n_atoms):
            if i < 10:
                names.append(f"C{i+1}")
            elif i < 20:
                names.append(f"O{i-9}")
            else:
                names.append(f"X{i+1}")
        
        return names
    
    def _create_state_metadata(
        self,
        state_id: int,
        state_energy: float,
        ligand_smiles: str,
        ligand_coords: np.ndarray,
        transition_network: TransitionNetwork
    ) -> Dict:
        """Create comprehensive metadata for the state"""
        
        return {
            "state_information": {
                "state_id": state_id,
                "binding_energy_kcal_mol": state_energy,
                "relative_energy": state_energy - transition_network.energies.min().item(),
                "ligand_smiles": ligand_smiles,
                "n_ligand_atoms": len(ligand_coords),
                "export_timestamp": str(np.datetime64('now'))
            },
            "structural_data": {
                "ligand_coordinates": ligand_coords.tolist(),
                "coordinate_units": "angstrom",
                "coordinate_precision": self.config.coordinate_precision,
                "ligand_positioned": self.config.position_ligand_near_protein,
                "binding_site_method": self.config.binding_site_detection
            },
            "analysis": {
                "binding_favorable": state_energy < -5.0,
                "interaction_strength": "strong" if state_energy < -8.0 else "moderate" if state_energy < -6.0 else "weak",
                "estimated_residence_time_ns": self._estimate_residence_time(state_energy)
            },
            "files": {
                "complex_pdb": f"complex_state_{state_id:03d}.pdb",
                "metadata": f"complex_state_{state_id:03d}_metadata.json"
            }
        }
    
    def _estimate_residence_time(self, binding_energy: float) -> float:
        """Rough estimate of residence time from binding energy"""
        
        # Very rough approximation: stronger binding = longer residence
        if binding_energy < -10:
            return 1000.0  # 1 μs
        elif binding_energy < -8:
            return 100.0   # 100 ns
        elif binding_energy < -6:
            return 10.0    # 10 ns
        else:
            return 1.0     # 1 ns
    
    def _generate_analysis_files(
        self,
        simulation_results: SimulationResults,
        transition_network: TransitionNetwork,
        ligand_smiles: str,
        output_path: Path
    ) -> List[str]:
        """Generate analysis summary files"""
        
        analysis_files = []
        
        # Comprehensive analysis summary
        summary_file = output_path / "analysis_summary.json"
        summary_data = {
            "ligand_analysis": {
                "smiles": ligand_smiles, 
                "n_transition_states": len(transition_network.positions),
                "energy_range_kcal_mol": {
                    "min": transition_network.energies.min().item(),
                    "max": transition_network.energies.max().item(),
                    "mean": transition_network.energies.mean().item()
                }
            },
            "positioning_info": {
                "ligand_positioned_near_protein": self.config.position_ligand_near_protein,
                "binding_site_detection_method": self.config.binding_site_detection,
                "target_ligand_protein_distance": self.config.ligand_protein_distance
            },
            "kinetic_analysis": {
                "binding_events": len(simulation_results.binding_times),
                "unbinding_events": len(simulation_results.unbinding_times),
                "mean_residence_time_ns": np.mean(simulation_results.binding_times) * 1e9 if len(simulation_results.binding_times) > 0 else 0
            },
            "files_generated": {
                "n_pdb_complexes": len(transition_network.positions),
                "n_metadata_files": len(transition_network.positions)
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        analysis_files.append(str(summary_file))
        
        return analysis_files
    
    def _generate_visualization_scripts(
        self,
        exported_files: List[Dict],
        output_path: Path
    ) -> List[str]:
        """Generate visualization scripts for PyMOL, VMD, etc."""
        
        viz_files = []
        
        if self.config.generate_pymol_script:
            pymol_script = self._create_pymol_script(exported_files, output_path)
            viz_files.append(pymol_script)
        
        if self.config.generate_vmd_script:
            vmd_script = self._create_vmd_script(exported_files, output_path)
            viz_files.append(vmd_script)
        
        return viz_files
    
    def _create_pymol_script(self, exported_files: List[Dict], output_path: Path) -> str:
        """Create enhanced PyMOL visualization script"""
        
        script_lines = [
            "# PandaKinetics Enhanced Visualization Script",
            "# Protein-Ligand Transition State Complexes",
            "# Generated by Enhanced Visualizer",
            "",
            "# Load all complexes",
        ]
        
        # Load all complexes
        for file_info in exported_files:
            pdb_name = Path(file_info["pdb_file"]).name
            state_id = file_info["state_id"]
            script_lines.append(f'load {pdb_name}, complex_{state_id:03d}')
        
        script_lines.extend([
            "",
            "# Color protein by B-factor (interaction strength)",
            "color gray, polymer",
            "spectrum b, blue_red, polymer",
            "",
            "# Color ligands by binding energy",
        ])
        
        # Color ligands by energy
        for file_info in exported_files:
            state_id = file_info["state_id"]
            energy = file_info["energy"]
            
            if energy < -10:
                color = "blue"
            elif energy < -8:
                color = "green"
            elif energy < -6:
                color = "yellow"
            else:
                color = "red"
            
            script_lines.append(f'color {color}, complex_{state_id:03d} and resn LIG')
        
        script_lines.extend([
            "",
            "# Display settings",
            "show cartoon, polymer",
            "show sticks, resn LIG",
            "show lines, (polymer within 4 of resn LIG)",
            "",
            "# Create binding site surface",
            "select binding_site, (polymer within 8 of resn LIG)",
            "show surface, binding_site",
            "set transparency, 0.3, binding_site",
            "",
            "# Animation setup",
            "set movie_panel, 1",
            f"mset 1 x{len(exported_files) * 10}",
            "",
            "# Energy labels",
        ])
        
        # Add energy labels
        for file_info in exported_files:
            state_id = file_info["state_id"]
            energy = file_info["energy"]
            script_lines.append(
                f'label complex_{state_id:03d} and resn LIG and name C1, "{energy:.1f}"'
            )
        
        script_lines.extend([
            "",
            "# Set view and zoom",
            "orient resn LIG",
            "zoom resn LIG, 8",
            "",
            "# Save session",
            "save transition_complexes.pse"
        ])
        
        script_file = output_path / "visualize_complexes.pml"
        with open(script_file, 'w') as f:
            f.write("\n".join(script_lines))
        
        logger.info(f"PyMOL script saved: {script_file}")
        return str(script_file)
    
    def _create_vmd_script(self, exported_files: List[Dict], output_path: Path) -> str:
        """Create VMD visualization script"""
        
        script_lines = [
            "# PandaKinetics VMD Visualization Script",
            "# Load transition state complexes",
            "",
        ]
        
        for file_info in exported_files:
            pdb_name = Path(file_info["pdb_file"]).name
            script_lines.append(f"mol new {pdb_name}")
        
        script_lines.extend([
            "",
            "# Set representations",
            "mol delrep 0 top",
            "mol representation NewCartoon",
            "mol selection protein",
            "mol addrep top",
            "",
            "mol representation Licorice",
            "mol selection resname LIG",
            "mol addrep top",
            "",
            "# Color by energy",
            "color Display Background white",
        ])
        
        script_file = output_path / "visualize_complexes.vmd"
        with open(script_file, 'w') as f:
            f.write("\n".join(script_lines))
        
        logger.info(f"VMD script saved: {script_file}")
        return str(script_file)


# Convenience function for CLI integration
def export_transition_complexes(
    results_dir: str,
    protein_pdb: str,
    ligand_smiles: str,
    output_dir: str = "transition_complexes",
    config: Optional[TransitionStateExportConfig] = None
) -> Dict[str, List[str]]:
    """
    Convenience function for exporting transition complexes
    
    Args:
        results_dir: Directory containing simulation results
        protein_pdb: Path to protein PDB file
        ligand_smiles: SMILES string of ligand
        output_dir: Output directory for complexes
        config: Export configuration
        
    Returns:
        Dictionary with lists of generated files
    """
    
    # Load results (this would be implemented based on PandaKinetics result format)
    # For now, using placeholder
    
    exporter = ProteinLigandComplexExporter(config)
    
    # This would load actual results from the results directory
    # simulation_results = load_simulation_results(results_dir)
    # transition_network = load_transition_network(results_dir)
    
    # return exporter.export_transition_complexes(
    #     simulation_results,
    #     transition_network,
    #     protein_pdb,
    #     ligand_smiles,
    #     output_dir
    # )
    
    # Placeholder return
    return {
        "message": "Integration point for PandaKinetics results loading",
        "results_dir": results_dir,
        "protein_pdb": protein_pdb,
        "ligand_smiles": ligand_smiles,
        "output_dir": output_dir
    }


def fix_existing_pdb_ligand_positioning(pdb_file: str, output_file: str = None) -> str:
    """
    Quick fix for existing PDB files where ligand is far from protein
    
    Args:
        pdb_file: Input PDB file with misplaced ligand
        output_file: Output file (if None, adds '_fixed' to input name)
        
    Returns:
        Path to fixed PDB file
    """
    
    if output_file is None:
        output_file = pdb_file.replace('.pdb', '_fixed.pdb')
    
    # Read the PDB file
    protein_atoms = []
    ligand_atoms = []
    other_lines = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                protein_atoms.append(line)
            elif line.startswith('HETATM') and ('LIG' in line or 'UNL' in line):
                ligand_atoms.append(line)
            else:
                other_lines.append(line)
    
    if not protein_atoms:
        logger.error("No protein atoms found in PDB file")
        return pdb_file
    
    if not ligand_atoms:
        logger.error("No ligand atoms found in PDB file")
        return pdb_file
    
    # Extract protein coordinates
    protein_coords = []
    for line in protein_atoms:
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            protein_coords.append([x, y, z])
        except ValueError:
            continue
    
    # Extract ligand coordinates
    ligand_coords = []
    for line in ligand_atoms:
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            ligand_coords.append([x, y, z])
        except ValueError:
            continue
    
    if not protein_coords or not ligand_coords:
        logger.error("Failed to extract coordinates")
        return pdb_file
    
    protein_coords = np.array(protein_coords)
    ligand_coords = np.array(ligand_coords)
    
    # Calculate centers
    protein_center = np.mean(protein_coords, axis=0)
    ligand_center = np.mean(ligand_coords, axis=0)
    
    logger.info(f"Protein center: {protein_center}")
    logger.info(f"Original ligand center: {ligand_center}")
    logger.info(f"Distance between centers: {np.linalg.norm(protein_center - ligand_center):.2f} Å")
    
    # Calculate translation to move ligand near protein
    translation = protein_center - ligand_center
    
    # Apply translation to ligand atoms and update PDB lines
    fixed_ligand_lines = []
    for i, line in enumerate(ligand_atoms):
        if i < len(ligand_coords):
            new_coords = ligand_coords[i] + translation
            # Reconstruct the line with new coordinates
            new_line = (line[:30] + 
                       f"{new_coords[0]:8.3f}{new_coords[1]:8.3f}{new_coords[2]:8.3f}" + 
                       line[54:])
            fixed_ligand_lines.append(new_line)
        else:
            fixed_ligand_lines.append(line)
    
    # Write the fixed PDB file
    with open(output_file, 'w') as f:
        # Write header and other lines
        for line in other_lines:
            if not line.startswith(('ATOM', 'HETATM')):
                f.write(line)
        
        # Write protein atoms
        f.writelines(protein_atoms)
        
        # Write fixed ligand atoms
        f.writelines(fixed_ligand_lines)
        
        # Write END if not already present
        if not any(line.startswith('END') for line in other_lines):
            f.write("END\n")
    
    # Verify the fix
    new_ligand_center = np.mean(ligand_coords + translation, axis=0)
    final_distance = np.linalg.norm(protein_center - new_ligand_center)
    
    logger.info(f"Fixed ligand center: {new_ligand_center}")
    logger.info(f"Final distance between centers: {final_distance:.2f} Å")
    logger.info(f"Fixed PDB saved to: {output_file}")
    
    return output_file