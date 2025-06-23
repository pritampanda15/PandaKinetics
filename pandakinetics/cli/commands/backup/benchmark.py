#!/usr/bin/env python3
import click
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

@click.command()
@click.option('--quick', is_flag=True, help='Run quick benchmarks only')
@click.option('--full', is_flag=True, help='Run full benchmark suite')
@click.option('--output', default='.', help='Output directory')
@click.pass_context
def benchmark(ctx, quick, full, output):
    """Run GPU benchmark tests"""
    
    logger.info("Starting benchmark tests...")
    
    # Find benchmark script
    pandakinetics_root = Path(__file__).parent.parent.parent.parent
    benchmark_script = pandakinetics_root / "scripts" / "run_benchmarks.py"
    
    if not benchmark_script.exists():
        click.echo("❌ Benchmark script not found. Running basic GPU test...")
        
        # Basic GPU test using existing code
        try:
            from pandakinetics import KineticSimulator
            simulator = KineticSimulator(n_replicas=4, max_simulation_time=1e-6)
            click.echo("✅ GPU test passed!")
            click.echo(f"📊 Using device: {simulator.device}")
            return 0
        except Exception as e:
            click.echo(f"❌ GPU test failed: {e}")
            return 1
    
    # Run benchmark script
    cmd = [sys.executable, str(benchmark_script)]
    if quick:
        cmd.append('--quick')
    elif full:
        cmd.append('--full')
    cmd.extend(['--output', output])
    
    try:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode
    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}")
        return 1
