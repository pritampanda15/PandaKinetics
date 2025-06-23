# =============================================================================
# pandakinetics/cli.py - Command Line Interface
# =============================================================================

import click
import sys
from pathlib import Path
from loguru import logger
import json

from .core import KineticSimulator
from .utils import check_gpu_availability
from .visualization import NetworkPlotter, KineticPlotter
from .cli.enhanced_commands import integrate_enhanced_commands

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--gpu', default=None, help='Specify GPU device (e.g., cuda:0)')
def cli(verbose, gpu):
    """PandaKinetics: Multi-Scale Structure-Kinetics Simulator for Drug Design"""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="{time} | {level} | {message}")
    
    # Check GPU availability
    if not check_gpu_availability():
        logger.error("GPU not available. PandaKinetics requires CUDA-capable GPU.")
        sys.exit(1)
    
    logger.info("PandaKinetics initialized")


@cli.command()
@click.option('--protein', '-p', required=True, help='Protein PDB file or PDB ID')
@click.option('--ligand', '-l', required=True, help='Ligand SMILES string')
@click.option('--output', '-o', default='results', help='Output directory')
@click.option('--n-poses', default=100, help='Number of docking poses')
@click.option('--simulation-time', default=1e-3, help='Simulation time (seconds)')
@click.option('--temperature', default=310.0, help='Temperature (K)')
@click.option('--n-replicas', default=16, help='Number of simulation replicas')
def predict(protein, ligand, output, n_poses, simulation_time, temperature, n_replicas):
    """Predict binding kinetics for a protein-ligand system"""
    
    logger.info(f"Starting kinetic prediction for {protein} + {ligand}")
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(exist_ok=True)
    
    # Initialize simulator
    simulator = KineticSimulator(
        temperature=temperature,
        n_replicas=n_replicas,
        n_poses=n_poses
    )
    
    try:
        # Run prediction
        results = simulator.predict_kinetics(
            protein_pdb=protein,
            ligand_smiles=ligand
        )
        
        # Save results
        results_dict = {
            'kon': results.kon,
            'koff': results.koff,
            'residence_time': results.residence_time,
            'binding_affinity': results.binding_affinity,
            'kinetic_selectivity': results.kinetic_selectivity,
            'pathway_analysis': results.pathway_analysis,
            'confidence_intervals': results.confidence_intervals
        }
        
        results_file = output_path / 'kinetic_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("KINETIC PREDICTION RESULTS")
        print("="*50)
        print(f"Association rate (kon): {results.kon:.2e} M⁻¹s⁻¹")
        print(f"Dissociation rate (koff): {results.koff:.2e} s⁻¹")
        print(f"Binding affinity (Kd): {results.binding_affinity:.2e} M")
        print(f"Residence time: {results.residence_time:.2e} s")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--results-file', '-r', required=True, help='Results JSON file')
@click.option('--output', '-o', default='plots', help='Output directory for plots')
def visualize(results_file, output):
    """Generate visualizations from kinetic results"""
    
    logger.info(f"Creating visualizations from {results_file}")
    
    # Load results
    try:
        with open(results_file, 'r') as f:
            results_dict = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(exist_ok=True)
    
    # Generate plots
    try:
        plotter = KineticPlotter()
        
        # Kinetic parameters plot
        plot_file = output_path / 'complete_kinetic_parameters.png'
        plotter.plot_kinetic_parameters(results_dict.get("kinetic_results", {}), str(plot_file))


        
        logger.info(f"Plots saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)


@cli.command()
def benchmark():
    """Run GPU benchmark tests"""
    
    logger.info("Running GPU benchmark...")
    
    from .utils.gpu_utils import GPUUtils
    
    # Memory info
    memory_info = GPUUtils.get_memory_info()
    print("\nGPU Memory Information:")
    print("-" * 30)
    for gpu_id, info in memory_info.items():
        print(f"{gpu_id}: {info['name']}")
        print(f"  Total: {info['total_memory']/1e9:.1f} GB")
        print(f"  Free: {info['free_memory']/1e9:.1f} GB")
        print(f"  Used: {info['allocated_memory']/1e9:.1f} GB")
    
    # Performance benchmark
    benchmark_results = GPUUtils.benchmark_gpu()
    print(f"\nPerformance Benchmark:")
    print("-" * 30)
    print(f"Device: {benchmark_results['device']}")
    print(f"Matrix multiplication ({benchmark_results['matrix_size']}x{benchmark_results['matrix_size']}):")
    print(f"  Time: {benchmark_results['total_time']:.3f} s")
    print(f"  Performance: {benchmark_results['gflops']:.1f} GFLOPS")


@cli.command()
@click.option('--config-file', '-c', help='Configuration file path')
def validate_config(config_file):
    """Validate configuration file"""
    
    if not config_file:
        logger.error("Configuration file required")
        sys.exit(1)
    
    logger.info(f"Validating configuration: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Basic validation
        required_fields = ['protein', 'ligand']
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                sys.exit(1)
        
        logger.info("Configuration file is valid")
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    cli()

cli = integrate_enhanced_commands(cli)
if __name__ == '__main__':
    main()
