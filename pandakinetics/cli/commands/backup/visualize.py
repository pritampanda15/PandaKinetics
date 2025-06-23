#!/usr/bin/env python3
import click
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@click.command()
@click.option('--results', '-r', required=True, help='Results directory')
@click.option('--output', '-o', default='visualization', help='Output directory')
@click.pass_context
def visualize(ctx, results, output):
    """Generate visualizations from kinetic results"""
    
    results_path = Path(results)
    output_path = Path(output)
    
    if not results_path.exists():
        click.echo(f"❌ Results directory not found: {results_path}")
        return 1
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Look for transition states
    transitions_dir = results_path / "transition_states"
    if transitions_dir.exists():
        pdb_files = list(transitions_dir.glob("*.pdb"))
        click.echo(f"🧬 Found {len(pdb_files)} transition states")
        
        # Generate PyMOL script
        script_lines = [
            "# PandaKinetics Visualization",
            ""
        ]
        
        for i, pdb_file in enumerate(sorted(pdb_files)):
            script_lines.append(f"load {pdb_file.name}, state_{i:03d}")
        
        script_lines.extend([
            "",
            "show sticks, all",
            "spectrum b, rainbow, all"
        ])
        
        script_file = output_path / "visualize.pml"
        with open(script_file, 'w') as f:
            f.write("\n".join(script_lines))
        
        click.echo(f"🎬 PyMOL script: {script_file}")
        click.echo(f"   To use: cd {transitions_dir} && pymol ../{script_file}")
    
    return 0
