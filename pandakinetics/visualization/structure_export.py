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
        
        # Export individual transition states
        exported_files = self._export_individual_states(
            transition_network, 
            protein_structure, 
            ligand_smiles, 
            output_path
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
                protein_data["atoms"].append(atom_data)
                protein_data["coordinates"].append([
                    atom_data["x"], atom_data["y"], atom_data["z"]
                ])
        
        protein_data["coordinates"] = np.array(protein_data["coordinates"])
        
        return protein_data
    
    def _parse_pdb_line(self, line: str) -> Dict:
        """Parse a PDB ATOM/HETATM line"""
        
        return {
            "record": line[:6].strip(),
            "serial": int(line[6:11].strip()),
            "name": line[12:16].strip(),
            "resname": line[17:20].strip(),
            "chain": line[21:22].strip(),
            "resseq": int(line[22:26].strip()),
            "x": float(line[30:38].strip()),
            "y": float(line[38:46].strip()),
            "z": float(line[46:54].strip()),
            "occupancy": float(line[54:60].strip()) if line[54:60].strip() else 1.0,
            "bfactor": float(line[60:66].strip()) if line[60:66].strip() else 0.0,
            "element": line[76:78].strip() if len(line) > 76 else "",
            "original_line": line
        }
    
    def _export_individual_states(
        self,
        transition_network: TransitionNetwork,
        protein_structure: Dict,
        ligand_smiles: str,
        output_path: Path
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
                output_path
            )
            exported_files.append(state_files)
            
        return exported_files
    
    def _export_single_state(
        self,
        state_id: int,
        transition_network: TransitionNetwork,
        protein_structure: Dict,
        ligand_smiles: str,
        output_path: Path
    ) -> Dict:
        """Export a single transition state"""
        
        # Get state data
        ligand_coords = transition_network.positions[state_id].cpu().numpy()
        state_energy = transition_network.energies[state_id].item()
        
        # Generate filenames
        pdb_file = output_path / f"complex_state_{state_id:03d}.pdb"
        metadata_file = output_path / f"complex_state_{state_id:03d}_metadata.json"
        
        # Create full complex PDB
        complex_pdb = self._create_complex_pdb(
            protein_structure,
            ligand_coords,
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
            ligand_coords,
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
        """Create a complete protein-ligand complex PDB"""
        
        lines = [
            "HEADER    PROTEIN-LIGAND COMPLEX                    " + "23-JUN-25   PKIN",
            f"TITLE     TRANSITION STATE {state_id} COMPLEX",
            f"REMARK   1 LIGAND SMILES: {ligand_smiles}",
            f"REMARK   2 BINDING ENERGY: {state_energy:.3f} KCAL/MOL",
            f"REMARK   3 STATE ID: {state_id}",
            f"REMARK   4 GENERATED BY PANDAKINETICS ENHANCED VISUALIZER",
            "REMARK   5",
            "REMARK   6 PROTEIN-LIGAND INTERACTIONS:",
        ]
        
        # Add interaction analysis
        interactions = self._analyze_interactions(protein_structure, ligand_coords)
        for i, interaction in enumerate(interactions[:10]):  # Top 10 interactions
            lines.append(
                f"REMARK   7 {interaction['protein_residue']} - "
                f"LIG{interaction['ligand_atom']:>3} "
                f"({interaction['distance']:.2f} A, {interaction['type']})"
            )
        
        lines.append("REMARK   8")
        
        # Add protein atoms (preserve original formatting)
        protein_atom_count = 0
        for atom in protein_structure["atoms"]:
            # Modify B-factor to reflect interaction strength
            bfactor = self._calculate_interaction_bfactor(atom, ligand_coords)
            
            modified_line = (
                f"{atom['record']:<6}{atom['serial']:>5} {atom['name']:>4} "
                f"{atom['resname']:>3} {atom['chain']}{atom['resseq']:>4}    "
                f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                f"{atom['occupancy']:6.2f}{bfactor:6.2f}           "
                f"{atom['element']:>2}"
            )
            lines.append(modified_line)
            protein_atom_count += 1
        
        # Add ligand atoms
        lines.append("REMARK   9 LIGAND ATOMS START")
        
        # Generate ligand atom names (simple approach)
        ligand_atom_names = self._generate_ligand_atom_names(len(ligand_coords))
        
        for i, coord in enumerate(ligand_coords):
            serial = protein_atom_count + i + 1
            atom_name = ligand_atom_names[i]
            
            lines.append(
                f"HETATM{serial:5d} {atom_name:>4} LIG L   1    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00{abs(state_energy):6.2f}           "
                f"{atom_name[0]:>2}"
            )
        
        # Add connectivity for ligand (if available)
        # This would require more sophisticated ligand analysis
        
        lines.extend([
            f"MASTER        0    0    0    0    0    0    0    0"
            f"{protein_atom_count:5d}    0{len(ligand_coords):5d}    0",
            "END"
        ])
        
        return "\n".join(lines) + "\n"
    
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
                "coordinate_precision": self.config.coordinate_precision
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
