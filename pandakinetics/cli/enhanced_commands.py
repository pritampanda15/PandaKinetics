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
from pandakinetics.visualization.structure_export import (
    ProteinLigandComplexExporter,
    TransitionStateExportConfig,
    export_transition_complexes
)
from pandakinetics.utils.config import Config
from pandakinetics.utils.gpu_utils import GPUUtils

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
    config = TransitionStateExportConfig(
        include_protein=include_protein,
        include_ligand=True,
        include_metadata=True,
        generate_pymol_script=generate_pymol,
        generate_vmd_script=generate_vmd,
        export_interactions=export_interactions,
        coordinate_precision=coordinate_precision
    )
    
    try:
        # Export transition complexes
        exported_files = export_transition_complexes(
            results_dir=str(results_path),
            protein_pdb=str(protein_path),
            ligand_smiles=ligand,
            output_dir=str(output_path),
            config=config
        )
        
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


# Enhanced predict command
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
              help='Automatically generate visualizations after prediction')
@click.option('--export-complexes/--no-export-complexes', default=False,
              help='Export protein-ligand transition state complexes')
@click.option('--gpu', default=None,
              help='GPU device to use (e.g. cuda:0, cuda:1)')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
def predict(ligand, protein, output, n_replicas, simulation_time, temperature,
            n_poses, auto_visualize, export_complexes, gpu, config, verbose):
    """
    Enhanced predict command with integrated visualization
    
    Examples:
    
    \b
    # Basic prediction
    pandakinetics predict -l "CCO" -p protein.pdb
    
    \b
    # Full workflow with visualization
    pandakinetics predict \\
        -l "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" \\
        -p cox2_structure.pdb \\
        -o ibuprofen_analysis \\
        --n-replicas 16 \\
        --auto-visualize \\
        --export-complexes \\
        --verbose
    
    \b
    # High-performance prediction
    pandakinetics predict -l "..." -p protein.pdb \\
        --gpu cuda:1 --n-replicas 32 --simulation-time 5e-6
    """
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    logger.info("Starting enhanced kinetic prediction...")
    
    # Validate inputs
    protein_path = Path(protein)
    output_path = Path(output)
    
    if not protein_path.exists():
        click.echo(f"Error: Protein file {protein_path} not found", err=True)
        return 1
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set GPU device
    if gpu:
        GPUUtils.set_device(gpu)
    
    # Load configuration
    if config:
        config_obj = Config.from_file(config)
    else:
        config_obj = Config.default()
    
    try:
        # Initialize simulator
        simulator = KineticSimulator(
            n_replicas=n_replicas,
            max_simulation_time=simulation_time,
            temperature=temperature
        )
        
        logger.info(f"Predicting kinetics for ligand: {ligand}")
        logger.info(f"Target protein: {protein_path}")
        logger.info(f"Output directory: {output_path}")
        
        # Run prediction using the method we discovered
        results = simulator.predict_kinetics(
            ligand_smiles=ligand,
            protein_pdb=str(protein_path),
            n_poses=n_poses,
            output_dir=str(output_path)
        )
        
        # Save results
        results_file = output_path / "kinetic_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        click.echo(f"\n✅ Kinetic prediction completed!")
        click.echo(f"📁 Results saved to: {output_path}")
        
        if hasattr(results, 'binding_events'):
            click.echo(f"🔗 Binding events: {results.binding_events}")
        if hasattr(results, 'unbinding_events'):
            click.echo(f"🔓 Unbinding events: {results.unbinding_events}")
        if hasattr(results, 'mean_residence_time'):
            click.echo(f"⏱️  Mean residence time: {results.mean_residence_time*1e9:.2f} ns")
        
        # Auto-visualization if requested
        if auto_visualize or export_complexes:
            logger.info("Running automatic visualization...")
            
            viz_output = output_path / "visualization"
            
            config = TransitionStateExportConfig(
                include_protein=export_complexes,
                generate_pymol_script=True,
                generate_vmd_script=False,
                export_interactions=True
            )
            
            viz_files = export_transition_complexes(
                results_dir=str(output_path),
                protein_pdb=str(protein_path),
                ligand_smiles=ligand,
                output_dir=str(viz_output),
                config=config
            )
            
            click.echo(f"🎬 Visualizations saved to: {viz_output}")
            
            if export_complexes:
                click.echo(f"🧬 Protein-ligand complexes exported")
                click.echo(f"   To visualize: cd {viz_output} && pymol visualize_complexes.pml")
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        return 1


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
