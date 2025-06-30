#!/usr/bin/env python3
"""
Enhanced CLI Commands for PandaKinetics
Files: 
- pandakinetics/cli/visualize.py (enhanced)
- pandakinetics/cli/predict.py (enhanced)
"""

import click
import json
import logging
from pathlib import Path
from typing import Optional

# PandaKinetics imports
from pandakinetics import KineticSimulator
from pandakinetics.utils.gpu_utils import GPUUtils

# Try to import visualization modules with fallbacks
try:
    from pandakinetics.visualization.structure_export import (
        ProteinLigandComplexExporter,
        TransitionStateExportConfig,
        export_transition_complexes
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    
    # Create dummy classes for compatibility
    class TransitionStateExportConfig:
        def __init__(self, **kwargs):
            pass
    
    def export_transition_complexes(*args, **kwargs):
        return {}

try:
    from pandakinetics.utils.config import Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    
    class Config:
        @staticmethod
        def from_file(path):
            return {}
        
        @staticmethod
        def default():
            return {}

logger = logging.getLogger(__name__)

# Enhanced visualize command
@click.command()
@click.option('--results', '-r', required=True, 
              help='Directory containing simulation results')
@click.option('--protein', '-p', required=True,
              help='Protein PDB file used in simulation')
@click.option('--ligand', '-l', required=True,
              help='Ligand SMILES string')
@click.option('--output', '-o', default='visualization_output',
              help='Output directory for visualizations')
@click.option('--include-protein/--no-protein', default=True,
              help='Include full protein in transition state PDBs')
@click.option('--generate-pymol/--no-pymol', default=True,
              help='Generate PyMOL visualization scripts')
@click.option('--generate-vmd/--no-vmd', default=False,
              help='Generate VMD visualization scripts')
@click.option('--export-interactions/--no-interactions', default=True,
              help='Export protein-ligand interaction analysis')
@click.option('--coordinate-precision', default=3, type=int,
              help='Decimal precision for coordinates')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
def visualize(results, protein, ligand, output, include_protein, 
              generate_pymol, generate_vmd, export_interactions,
              coordinate_precision, verbose):
    """
    Enhanced visualization command with protein-ligand complex export
    
    Examples:
    
    \b
    # Basic usage
    pandakinetics visualize -r results/ -p protein.pdb -l "CCO"
    
    \b
    # Full protein-ligand complexes with PyMOL script
    pandakinetics visualize -r results/ -p cox2.pdb \\
        -l "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \\
        --output ibuprofen_complexes --generate-pymol
    
    \b
    # Ligand-only with VMD support
    pandakinetics visualize -r results/ -p protein.pdb -l "CCO" \\
        --no-protein --generate-vmd --export-interactions
    """
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    logger.info("Starting enhanced visualization...")
    
    # Validate inputs
    results_path = Path(results)
    protein_path = Path(protein)
    output_path = Path(output)
    
    if not results_path.exists():
        click.echo(f"Error: Results directory {results_path} not found", err=True)
        return 1
    
    if not protein_path.exists():
        click.echo(f"Error: Protein file {protein_path} not found", err=True)
        return 1
    
    # Create export configuration
    if VISUALIZATION_AVAILABLE:
        config = TransitionStateExportConfig(
            include_protein=include_protein,
            include_ligand=True,
            include_metadata=True,
            generate_pymol_script=generate_pymol,
            generate_vmd_script=generate_vmd,
            export_interactions=export_interactions,
            coordinate_precision=coordinate_precision
        )
    else:
        config = TransitionStateExportConfig()
    
    try:
        # Export transition complexes
        if VISUALIZATION_AVAILABLE:
            exported_files = export_transition_complexes(
                results_dir=str(results_path),
                protein_pdb=str(protein_path),
                ligand_smiles=ligand,
                output_dir=str(output_path),
                config=config
            )
        else:
            click.echo("⚠️  Visualization features not available. Install visualization dependencies.")
            exported_files = {}
        
        # Print summary
        click.echo(f"\n✅ Visualization export completed!")
        click.echo(f"📁 Output directory: {output_path}")
        
        if 'pdb_complexes' in exported_files:
            click.echo(f"🧬 PDB complexes: {len(exported_files['pdb_complexes'])}")
        
        if 'visualization_scripts' in exported_files:
            click.echo(f"📊 Visualization scripts: {len(exported_files['visualization_scripts'])}")
        
        # Instructions for visualization
        if generate_pymol:
            pymol_script = output_path / "visualize_complexes.pml"
            if pymol_script.exists():
                click.echo(f"\n🎬 To visualize in PyMOL:")
                click.echo(f"   cd {output_path}")
                click.echo(f"   pymol visualize_complexes.pml")
        
        if generate_vmd:
            vmd_script = output_path / "visualize_complexes.vmd"
            if vmd_script.exists():
                click.echo(f"\n🎬 To visualize in VMD:")
                click.echo(f"   cd {output_path}")
                click.echo(f"   vmd -e visualize_complexes.vmd")
        
        return 0
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        return 1


# Import the enhanced predict command with Boltz-2 support
from .commands.predict import predict


# Complete CLI group with enhanced commands
@click.group()
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose logging')
@click.option('--gpu', help='Specify GPU device (e.g., cuda:0)')
@click.pass_context
def cli(ctx, verbose, gpu):
    """
    PandaKinetics: Multi-Scale Structure-Kinetics Simulator for Drug Design
    Enhanced with comprehensive protein-ligand complex visualization
    """
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Set global options
    ctx.obj['verbose'] = verbose
    ctx.obj['gpu'] = gpu
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    if gpu:
        GPUUtils.set_device(gpu)

# Add commands to the group
cli.add_command(predict)
cli.add_command(visualize)

# Integration function for existing CLI
def integrate_enhanced_commands(existing_cli):
    """
    Integrate enhanced commands into existing PandaKinetics CLI
    
    Usage in main CLI file:
    
    from pandakinetics.cli.enhanced_commands import integrate_enhanced_commands
    integrate_enhanced_commands(cli)
    """
    
    # Remove existing commands if they exist
    if 'predict' in existing_cli.commands:
        del existing_cli.commands['predict']
    if 'visualize' in existing_cli.commands:
        del existing_cli.commands['visualize']
    
    # Add enhanced commands
    existing_cli.add_command(predict)
    existing_cli.add_command(visualize)
    
    return existing_cli

# Example usage script
def create_example_usage():
    """Generate example usage script"""
    
    example_script = '''#!/bin/bash
# PandaKinetics Enhanced Workflow Examples

# Example 1: Basic ibuprofen-COX2 analysis
pandakinetics predict \\
    --ligand "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \\
    --protein cox2_structure.pdb \\
    --output ibuprofen_cox2_analysis \\
    --n-replicas 16 \\
    --auto-visualize \\
    --export-complexes \\
    --verbose

# Example 2: High-throughput screening setup
for compound in compounds.txt; do
    pandakinetics predict \\
        -l "$compound" \\
        -p target_protein.pdb \\
        -o "screening_results/$(echo $compound | tr '/' '_')" \\
        --n-replicas 8 \\
        --gpu cuda:0
done

# Example 3: Visualization only from existing results
pandakinetics visualize \\
    --results ibuprofen_cox2_analysis \\
    --protein cox2_structure.pdb \\
    --ligand "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \\
    --output enhanced_visualization \\
    --generate-pymol \\
    --generate-vmd \\
    --export-interactions

# Example 4: Fragment-based drug design workflow
pandakinetics predict \\
    -l "c1ccccc1" \\
    -p protein.pdb \\
    -o fragment_analysis \\
    --n-poses 100 \\
    --export-complexes \\
    --auto-visualize

echo "Enhanced PandaKinetics workflows completed!"
'''
    
    with open('pandakinetics_enhanced_examples.sh', 'w') as f:
        f.write(example_script)
    
    print("Example usage script created: pandakinetics_enhanced_examples.sh")

if __name__ == "__main__":
    cli()
