#!/usr/bin/env python3
"""
Complete PandaKinetics analysis example: Ibuprofen binding to COX-2
Demonstrates transition state analysis and PDB output generation
"""

from pandakinetics import KineticSimulator
import json
from pathlib import Path

def analyze_ibuprofen_cox2():
    """
    Complete kinetic analysis of ibuprofen binding to COX-2
    """
    
    # Initialize simulator
    simulator = KineticSimulator(
        n_replicas=16,
        max_simulation_time=1e-6,  # 1 microsecond
        temperature=310.15,  # Body temperature
        output_dir="ibuprofen_cox2_analysis"
    )
    
    # Ibuprofen SMILES
    ibuprofen_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    
    print("Analyzing Ibuprofen binding to COX-2...")
    print(f"Ligand SMILES: {ibuprofen_smiles}")
    
    # Run full kinetic analysis
    results = simulator.analyze_ligand(
        ligand_smiles=ibuprofen_smiles,
        target_protein="cox2_structure.pdb",  # Your COX-2 structure
        binding_site_residues=[120, 348, 384, 523],  # Known COX-2 binding site
        generate_transition_pdbs=True  # Key parameter for PDB output
    )
    
    return results

def export_transition_complexes(results, output_dir="transition_complexes"):
    """
    Export all transition states as PDB complexes with full metadata
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nExporting transition states to {output_dir}/")
    
    # Get transition network
    network = results.transition_network
    
    # Export each state as a PDB complex
    exported_states = []
    
    for state_id, state in enumerate(network.states):
        
        # Generate complex PDB file
        pdb_filename = output_path / f"ibuprofen_cox2_state_{state_id:03d}.pdb"
        
        # Create PDB with full metadata
        pdb_content = generate_complex_pdb(state, results.ligand_smiles)
        
        with open(pdb_filename, 'w') as f:
            f.write(pdb_content)
        
        # Create detailed metadata JSON
        metadata = {
            "state_id": state_id,
            "ligand_info": {
                "smiles": results.ligand_smiles,
                "name": "Ibuprofen",
                "molecular_weight": 206.28,
                "formal_charge": -1  # At physiological pH
            },
            "energetics": {
                "binding_energy_kcal_mol": state.binding_energy,
                "relative_energy_from_bound": state.energy - network.bound_state_energy,
                "probability": state.equilibrium_probability,
                "free_energy_barrier": state.transition_barrier if hasattr(state, 'transition_barrier') else None
            },
            "kinetics": {
                "residence_time_seconds": state.mean_residence_time,
                "association_rate_m1s1": state.kon if hasattr(state, 'kon') else None,
                "dissociation_rate_s1": state.koff if hasattr(state, 'koff') else None,
                "transition_rates": {
                    f"to_state_{j}": rate for j, rate in enumerate(state.transition_rates)
                }
            },
            "structural_analysis": {
                "protein_contacts": [
                    {
                        "residue": contact.residue_name + str(contact.residue_number),
                        "atom": contact.atom_name,
                        "distance_angstrom": contact.distance,
                        "interaction_type": contact.interaction_type
                    }
                    for contact in state.protein_contacts
                ],
                "hydrogen_bonds": state.hydrogen_bonds,
                "hydrophobic_contacts": state.hydrophobic_contacts,
                "rmsd_from_crystallographic": state.rmsd_crystal if hasattr(state, 'rmsd_crystal') else None,
                "ligand_conformation": {
                    "torsion_angles": state.ligand_torsions,
                    "ring_conformations": state.ring_conformations
                }
            },
            "pathway_analysis": {
                "pathway_step": state_id,
                "is_binding_intermediate": state.is_binding_intermediate,
                "is_transition_state": state.is_transition_state,
                "preceding_states": state.preceding_states,
                "following_states": state.following_states
            }
        }
        
        # Save metadata
        json_filename = output_path / f"ibuprofen_cox2_state_{state_id:03d}_metadata.json"
        with open(json_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        exported_states.append({
            "pdb_file": str(pdb_filename),
            "metadata_file": str(json_filename),
            "binding_energy": state.binding_energy,
            "residence_time": state.mean_residence_time
        })
        
        print(f"  State {state_id:03d}: ΔG = {state.binding_energy:6.2f} kcal/mol, "
              f"τ = {state.mean_residence_time*1e9:6.1f} ns")
    
    return exported_states

def generate_complex_pdb(state, ligand_smiles):
    """
    Generate PDB file for protein-ligand complex
    """
    
    pdb_lines = []
    
    # Header information
    pdb_lines.extend([
        "HEADER    PROTEIN-LIGAND COMPLEX                    " + "23-JUN-25   PKIN",
        "TITLE     IBUPROFEN-COX2 BINDING TRANSITION STATE",
        f"REMARK   1 LIGAND SMILES: {ligand_smiles}",
        f"REMARK   2 BINDING ENERGY: {state.binding_energy:.3f} KCAL/MOL",
        f"REMARK   3 STATE PROBABILITY: {state.equilibrium_probability:.6f}",
        f"REMARK   4 RESIDENCE TIME: {state.mean_residence_time*1e9:.2f} NS",
        "REMARK   5 GENERATED BY PANDAKINETICS",
        "REMARK   6",
        "REMARK   7 KEY INTERACTIONS:",
    ])
    
    # Add interaction remarks
    for contact in state.protein_contacts[:5]:  # Top 5 contacts
        pdb_lines.append(
            f"REMARK   8 {contact.residue_name}{contact.residue_number:>4} "
            f"{contact.atom_name:>4} - LIG {contact.ligand_atom:>4} "
            f"({contact.distance:.2f} A, {contact.interaction_type})"
        )
    
    pdb_lines.append("REMARK   9")
    
    # Protein coordinates
    for i, (atom_name, coords) in enumerate(zip(state.protein_atom_names, state.protein_coordinates)):
        pdb_lines.append(
            f"ATOM  {i+1:5d} {atom_name:>4} {state.residue_names[i]:>3} A{state.residue_numbers[i]:>4}    "
            f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00 20.00           "
            f"{atom_name[0]:>2}"
        )
    
    # Ligand coordinates  
    pdb_lines.append("HETATM" + " "*74)  # Separator
    
    for i, (atom_name, coords) in enumerate(zip(state.ligand_atom_names, state.ligand_coordinates)):
        pdb_lines.append(
            f"HETATM{len(state.protein_coordinates)+i+1:5d} {atom_name:>4} IBU L   1    "
            f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00 30.00           "
            f"{atom_name[0]:>2}"
        )
    
    # Connectivity (CONECT records for ligand)
    for bond in state.ligand_bonds:
        atom1_idx = len(state.protein_coordinates) + bond.atom1 + 1
        atom2_idx = len(state.protein_coordinates) + bond.atom2 + 1
        pdb_lines.append(f"CONECT{atom1_idx:5d}{atom2_idx:5d}")
    
    pdb_lines.extend([
        "MASTER        0    0    0    0    0    0    0    0 " + 
        f"{len(state.protein_coordinates):4d}    0 " +
        f"{len(state.ligand_coordinates):4d}    0",
        "END"
    ])
    
    return "\n".join(pdb_lines) + "\n"

def generate_analysis_summary(results, exported_states):
    """
    Generate comprehensive analysis summary
    """
    
    summary = {
        "ligand_analysis": {
            "name": "Ibuprofen",
            "smiles": results.ligand_smiles,
            "target": "COX-2",
            "simulation_time_us": results.simulation_time * 1e6
        },
        "binding_kinetics": {
            "mean_residence_time_ns": results.mean_residence_time * 1e9,
            "association_rate_m1s1": results.kon,
            "dissociation_rate_s1": results.koff,
            "binding_affinity_kd_um": (results.koff / results.kon) * 1e6,
            "total_binding_events": len(results.binding_times),
            "total_unbinding_events": len(results.unbinding_times)
        },
        "pathway_analysis": {
            "number_of_states": len(exported_states),
            "binding_pathway_length": results.mean_pathway_length,
            "rate_limiting_step": results.rate_limiting_transition,
            "major_intermediates": [
                state for state in exported_states 
                if state["residence_time"] > results.mean_residence_time * 0.1
            ]
        },
        "structural_insights": {
            "key_binding_residues": results.critical_residues,
            "allosteric_effects": results.allosteric_coupling if hasattr(results, 'allosteric_coupling') else None,
            "conformational_changes": results.protein_conformational_changes
        },
        "drug_design_implications": {
            "selectivity_hotspots": results.selectivity_determining_residues,
            "optimization_targets": results.optimization_recommendations,
            "predicted_mutations_impact": results.mutation_sensitivity
        }
    }
    
    # Save comprehensive summary
    with open("ibuprofen_cox2_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def create_pymol_visualization_script(exported_states):
    """
    Generate PyMOL script for pathway visualization
    """
    
    script_lines = [
        "# PandaKinetics Binding Pathway Visualization",
        "# Ibuprofen binding to COX-2",
        "",
        "# Load all transition states",
    ]
    
    # Load all PDB files
    for i, state in enumerate(exported_states):
        pdb_file = Path(state["pdb_file"]).name
        script_lines.append(f'load {pdb_file}, state_{i:03d}')
    
    script_lines.extend([
        "",
        "# Color protein by state energy",
        "# Blue = low energy (stable), Red = high energy (unstable)",
    ])
    
    # Color by binding energy
    for i, state in enumerate(exported_states):
        energy = state["binding_energy"]
        if energy < -8:  # Strong binding
            color = "blue"
        elif energy < -5:  # Moderate binding  
            color = "green"
        elif energy < -2:  # Weak binding
            color = "yellow"
        else:  # Unfavorable
            color = "red"
        
        script_lines.append(f'color {color}, state_{i:03d} and polymer')
    
    script_lines.extend([
        "",
        "# Show ligand as sticks, protein as cartoon",
        "show cartoon, polymer",
        "show sticks, resn IBU",
        "",
        "# Highlight binding site residues",
        "show sticks, (resi 120+348+384+523) and polymer",
        "color cyan, (resi 120+348+384+523) and polymer",
        "",
        "# Create binding pathway movie",
        "set movie_panel, 1",
        "mset 1 x100",  # 100 frames
        "",
        "# Animation showing binding pathway",
        "python",
        "for i in range(100):",
        f"    state_id = i * {len(exported_states)} // 100",
        "    cmd.frame(i+1)",
        "    cmd.hide('everything')",
        f"    cmd.show('cartoon', f'state_{{state_id:03d}} and polymer')",
        f"    cmd.show('sticks', f'state_{{state_id:03d}} and resn IBU')",
        "python end",
        "",
        "# Set view",
        "orient resn IBU",
        "zoom resn IBU, 8",
        "",
        "# Labels for key states",
        "label state_000 and resn IBU and name C1, 'Unbound'",
        f"label state_{len(exported_states)-1:03d} and resn IBU and name C1, 'Bound'",
    ])
    
    # Save PyMOL script
    with open("visualize_ibuprofen_pathway.pml", 'w') as f:
        f.write("\n".join(script_lines))
    
    print(f"PyMOL visualization script saved: visualize_ibuprofen_pathway.pml")

def main():
    """
    Complete analysis workflow
    """
    
    print("PandaKinetics: Ibuprofen-COX2 Kinetic Analysis")
    print("=" * 50)
    
    # Run kinetic analysis
    results = analyze_ibuprofen_cox2()
    
    # Export transition state PDB complexes
    exported_states = export_transition_complexes(results)
    
    # Generate analysis summary
    summary = generate_analysis_summary(results, exported_states)
    
    # Create visualization script
    create_pymol_visualization_script(exported_states)
    
    # Print key results
    print(f"\nKey Results:")
    print(f"  Residence time: {results.mean_residence_time*1e9:.1f} ns")
    print(f"  Binding affinity (Kd): {(results.koff/results.kon)*1e6:.2f} μM")
    print(f"  Number of binding intermediates: {len(exported_states)}")
    print(f"  Rate-limiting step: State {results.rate_limiting_transition}")
    
    print(f"\nFiles generated:")
    print(f"  - {len(exported_states)} PDB complex files")
    print(f"  - {len(exported_states)} metadata JSON files")
    print(f"  - Analysis summary: ibuprofen_cox2_analysis_summary.json")
    print(f"  - PyMOL visualization: visualize_ibuprofen_pathway.pml")
    
    print(f"\nTo visualize in PyMOL:")
    print(f"  pymol visualize_ibuprofen_pathway.pml")

if __name__ == "__main__":
    main()
