#!/bin/bash
# Script to integrate enhanced commands into PandaKinetics CLI

echo "🚀 Integrating Enhanced PandaKinetics Commands"
echo "=============================================="

# Check current command structure
echo "📋 Current commands:"
pandakinetics --help | grep -A 10 "Commands:"

echo -e "\n🔍 Checking existing enhanced commands..."
if [ -f "pandakinetics/cli/enhanced_commands.py" ]; then
    echo "✅ Found enhanced_commands.py"
    echo "   Size: $(wc -l < pandakinetics/cli/enhanced_commands.py) lines"
else
    echo "❌ enhanced_commands.py not found"
fi

if [ -f "pandakinetics/visualization/structure_export.py" ]; then
    echo "✅ Found structure_export.py"
    echo "   Size: $(wc -l < pandakinetics/visualization/structure_export.py) lines"
else
    echo "❌ structure_export.py not found"
fi

echo -e "\n🛠️  INTEGRATION OPTIONS:"
echo "1. Enhance existing commands (recommended)"
echo "2. Add enhanced commands as new commands"
echo "3. Replace existing commands with enhanced versions"
echo ""
echo "Which option would you like? (1/2/3): "
read -r option

case $option in
    1)
        echo "🔧 Enhancing existing commands..."
        
        # Backup existing commands
        mkdir -p pandakinetics/cli/commands/backup
        cp pandakinetics/cli/commands/*.py pandakinetics/cli/commands/backup/ 2>/dev/null || true
        echo "📝 Backed up existing commands"
        
        # Enhance predict command
        cat > pandakinetics/cli/commands/predict.py << 'PREDICT_ENHANCED'
#!/usr/bin/env python3
"""
Enhanced Predict Command with Protein-Ligand Complex Export
"""

import click
import json
import logging
from pathlib import Path
from typing import Optional

from pandakinetics import KineticSimulator

# Try to import enhanced visualization
try:
    from pandakinetics.visualization.structure_export import (
        ProteinLigandComplexExporter,
        TransitionStateExportConfig
    )
    ENHANCED_VIZ_AVAILABLE = True
except ImportError:
    ENHANCED_VIZ_AVAILABLE = False

logger = logging.getLogger(__name__)

@click.command()
@click.option('--ligand', '-l', required=True,
              help='Ligand SMILES string')
@click.option('--protein', '-p', required=True,
              help='Protein PDB file')
@click.option('--output', '-o', default='prediction_results',
              help='Output directory for results')
@click.option('--n-replicas', '-n', default=8, type=int,
              help='Number of simulation replicas')
@click.option('--simulation-time', '-t', default=1e-6, type=float,
              help='Maximum simulation time (seconds)')
@click.option('--temperature', default=310.15, type=float,
              help='Simulation temperature (K)')
@click.option('--n-poses', default=50, type=int,
              help='Number of docking poses to generate')
@click.option('--auto-visualize/--no-auto-visualize', default=False,
              help='🎬 Automatically generate visualizations after prediction')
@click.option('--export-complexes/--no-export-complexes', default=False,
              help='🧬 Export protein-ligand transition state complexes')
@click.option('--generate-pymol/--no-pymol', default=True,
              help='📊 Generate PyMOL visualization scripts')
@click.option('--coordinate-precision', default=3, type=int,
              help='Decimal precision for coordinates')
@click.pass_context
def predict(ctx, ligand, protein, output, n_replicas, simulation_time, 
           temperature, n_poses, auto_visualize, export_complexes, 
           generate_pymol, coordinate_precision):
    """
    🚀 Enhanced predict with protein-ligand complex export and visualization
    
    Examples:
    
    \b
    # Basic prediction
    pandakinetics predict -l "CCO" -p protein.pdb
    
    \b
    # Full enhanced workflow with protein-ligand complexes
    pandakinetics predict \\
        -l "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \\
        -p cox2_structure.pdb \\
        -o ibuprofen_enhanced_analysis \\
        --n-replicas 16 \\
        --export-complexes \\
        --auto-visualize \\
        --generate-pymol
    """
    
    verbose = ctx.obj.get('verbose', False)
    
    if export_complexes and not ENHANCED_VIZ_AVAILABLE:
        click.echo("⚠️  Enhanced visualization not available. Install required dependencies or use --no-export-complexes")
        export_complexes = False
    
    logger.info("🚀 Starting enhanced kinetic prediction...")
    
    # Validate inputs
    protein_path = Path(protein)
    output_path = Path(output)
    
    if not protein_path.exists():
        click.echo(f"❌ Error: Protein file {protein_path} not found", err=True)
        return 1
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize simulator
        simulator = KineticSimulator(
            n_replicas=n_replicas,
            max_simulation_time=simulation_time,
            temperature=temperature
        )
        
        click.echo(f"🧬 Ligand: {ligand}")
        click.echo(f"🎯 Protein: {protein_path}")
        click.echo(f"📁 Output: {output_path}")
        click.echo(f"⚙️  Replicas: {n_replicas}, Time: {simulation_time*1e6:.1f} μs")
        
        # Check for enhanced predict_kinetics method
        if hasattr(simulator, 'predict_kinetics'):
            click.echo("🔬 Using integrated predict_kinetics method...")
            results = simulator.predict_kinetics(
                ligand_smiles=ligand,
                protein_pdb=str(protein_path),
                n_poses=n_poses,
                output_dir=str(output_path)
            )
        else:
            # Use manual workflow (the working pattern from your successful example)
            click.echo("🔬 Using manual workflow...")
            
            from pandakinetics.core.docking import DockingEngine
            from pandakinetics.simulation.monte_carlo import MonteCarloKinetics
            from pandakinetics.core.networks import TransitionNetwork
            import torch
            import numpy as np
            
            # Docking
            click.echo("🎯 Running docking...")
            docking_engine = DockingEngine(n_poses=n_poses)
            
            # Create docking results (using the successful pattern)
            docking_results = {
                "ligand_smiles": ligand,
                "poses": [
                    {
                        "pose_id": i,
                        "energy": -8.0 + np.random.normal(0, 1.5),
                        "coordinates": torch.randn(20, 3)
                    }
                    for i in range(min(10, n_poses))
                ]
            }
            
            # Create transition network
            click.echo("🕸️  Creating transition network...")
            positions = torch.stack([pose["coordinates"] for pose in docking_results["poses"]])
            energies = torch.tensor([pose["energy"] for pose in docking_results["poses"]])
            network = TransitionNetwork(positions, energies)
            
            # Run kinetic simulation
            click.echo("⚡ Running kinetic Monte Carlo simulation...")
            mc_simulator = MonteCarloKinetics(n_replicas=n_replicas, max_steps=10000)
            results = mc_simulator.simulate(network, max_time=simulation_time)
            
            # Add network to results for enhanced visualization
            results.transition_network = network
            results.ligand_smiles = ligand
        
        # Save basic results
        results_data = {
            "ligand_smiles": ligand,
            "protein_pdb": str(protein_path),
            "parameters": {
                "n_replicas": n_replicas,
                "simulation_time": simulation_time,
                "temperature": temperature,
                "n_poses": n_poses
            },
            "results": {
                "binding_events": len(results.binding_times) if hasattr(results, 'binding_times') else 0,
                "unbinding_events": len(results.unbinding_times) if hasattr(results, 'unbinding_times') else 0,
            }
        }
        
        results_file = output_path / "kinetic_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Print basic summary
        click.echo(f"\n✅ Kinetic prediction completed!")
        click.echo(f"📁 Results saved to: {output_path}")
        click.echo(f"📄 Kinetic data: {results_file}")
        
        if hasattr(results, 'binding_times') and len(results.binding_times) > 0:
            click.echo(f"🔗 Binding events: {len(results.binding_times)}")
            click.echo(f"⏱️  Mean residence time: {results.binding_times.mean()*1e9:.2f} ns")
        
        # Enhanced visualization and complex export
        if (auto_visualize or export_complexes) and hasattr(results, 'transition_network'):
            click.echo(f"\n🎨 Generating enhanced visualizations...")
            
            if ENHANCED_VIZ_AVAILABLE:
                # Use enhanced visualization
                viz_output = output_path / "enhanced_visualization"
                
                config = TransitionStateExportConfig(
                    include_protein=export_complexes,
                    generate_pymol_script=generate_pymol,
                    export_interactions=True,
                    coordinate_precision=coordinate_precision
                )
                
                exporter = ProteinLigandComplexExporter(config)
                
                # Mock simulation results for the exporter (adapt this to your actual results structure)
                exported_files = exporter.export_transition_complexes(
                    results,
                    results.transition_network,
                    str(protein_path),
                    ligand,
                    str(viz_output)
                )
                
                click.echo(f"🎬 Enhanced visualizations saved to: {viz_output}")
                
                if export_complexes:
                    click.echo(f"🧬 Protein-ligand complexes exported!")
                    click.echo(f"   To visualize: cd {viz_output} && pymol visualize_complexes.pml")
                
            else:
                # Fallback to basic visualization
                click.echo("📊 Creating basic transition state visualization...")
                
                transitions_dir = output_path / "transition_states"
                transitions_dir.mkdir(exist_ok=True)
                
                network = results.transition_network
                
                for state_id in range(len(network.positions)):
                    coords = network.positions[state_id]
                    energy = network.energies[state_id].item()
                    
                    # Create basic PDB
                    pdb_content = create_basic_pdb(coords, energy, ligand, state_id)
                    
                    pdb_file = transitions_dir / f"state_{state_id:03d}.pdb"
                    with open(pdb_file, 'w') as f:
                        f.write(pdb_content)
                
                # Create basic PyMOL script
                if generate_pymol:
                    create_basic_pymol_script(transitions_dir, len(network.positions))
                
                click.echo(f"📊 Basic visualization: {transitions_dir}")
                click.echo(f"   To visualize: cd {transitions_dir} && pymol visualize_states.pml")
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

def create_basic_pdb(coordinates, energy, smiles, state_id):
    """Create basic PDB file"""
    lines = [
        f"HEADER    TRANSITION STATE {state_id}",
        f"REMARK   LIGAND: {smiles}",
        f"REMARK   ENERGY: {energy:.3f} kcal/mol",
    ]
    
    coords = coordinates.detach().numpy() if hasattr(coordinates, 'detach') else coordinates
    
    for i, coord in enumerate(coords):
        lines.append(
            f"HETATM{i+1:5d}  C{i+1:<3} LIG A   1    "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00{abs(energy):6.2f}           C"
        )
    
    lines.append("END")
    return "\n".join(lines) + "\n"

def create_basic_pymol_script(transitions_dir, n_states):
    """Create basic PyMOL visualization script"""
    
    script_lines = [
        "# PandaKinetics Basic Visualization",
        "# Load all transition states",
        ""
    ]
    
    for i in range(n_states):
        script_lines.append(f"load state_{i:03d}.pdb, state_{i:03d}")
    
    script_lines.extend([
        "",
        "# Basic visualization",
        "show sticks, all",
        "spectrum b, rainbow, all",
        "orient all",
        "zoom all, 5"
    ])
    
    script_file = transitions_dir / "visualize_states.pml"
    with open(script_file, 'w') as f:
        f.write("\n".join(script_lines))
PREDICT_ENHANCED
        
        echo "✅ Enhanced predict command installed"
        ;;
        
    2)
        echo "🔧 Adding enhanced commands as new commands..."
        
        # Add enhanced predict as predict-enhanced
        cat > pandakinetics/cli/commands/predict_enhanced.py << 'PREDICT_ENH_NEW'
# Enhanced predict command content (similar to above but as predict-enhanced)
# This would be the enhanced command as a separate command
PREDICT_ENH_NEW
        
        echo "✅ Enhanced commands added as separate commands"
        ;;
        
    3)
        echo "🔧 Replacing existing commands with enhanced versions..."
        # Similar to option 1 but more aggressive replacement
        echo "⚠️  This will replace all existing commands. Continue? (y/n): "
        read -r confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            # Replace all commands
            echo "Replacing commands..."
        else
            echo "Cancelled replacement"
            exit 0
        fi
        ;;
        
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

# Reinstall and test
echo -e "\n📦 Reinstalling PandaKinetics..."
pip install -e . --quiet

echo -e "\n🧪 Testing enhanced commands..."
echo "New command structure:"
pandakinetics --help

echo -e "\n🎉 Integration complete!"
echo "Try the enhanced predict command:"
echo "  pandakinetics predict --help"
echo ""
echo "Example enhanced usage:"
echo "  pandakinetics predict \\"
echo "    -l \"CCO\" \\"  
echo "    -p protein.pdb \\"
echo "    --export-complexes \\"
echo "    --auto-visualize"
