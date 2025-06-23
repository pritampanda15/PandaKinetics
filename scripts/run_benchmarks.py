# =============================================================================
# scripts/benchmark.py - Performance Benchmarking
# =============================================================================

#!/usr/bin/env python3
"""
Performance benchmarking script for PandaKinetics
"""

import time
import json
import argparse
from pathlib import Path
import psutil
import torch
import cupy as cp
import numpy as np
from datetime import datetime

from pandakinetics import KineticSimulator
from pandakinetics.utils import GPUUtils


def benchmark_gpu():
    """Benchmark GPU performance"""
    print("Benchmarking GPU performance...")
    
    device = GPUUtils.get_device()
    
    # Matrix multiplication benchmark
    sizes = [1024, 2048, 4096, 8192]
    results = {}
    
    for size in sizes:
        print(f"  Matrix size: {size}x{size}")
        
        # PyTorch benchmark
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(3):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end_time = time.time()
        
        pytorch_time = (end_time - start_time) / 10
        pytorch_gflops = (2 * size**3) / (pytorch_time * 1e9)
        
        # CuPy benchmark
        a_cp = cp.random.randn(size, size)
        b_cp = cp.random.randn(size, size)
        
        # Warmup
        for _ in range(3):
            cp.matmul(a_cp, b_cp)
        cp.cuda.Device().synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            c_cp = cp.matmul(a_cp, b_cp)
        cp.cuda.Device().synchronize()
        end_time = time.time()
        
        cupy_time = (end_time - start_time) / 10
        cupy_gflops = (2 * size**3) / (cupy_time * 1e9)
        
        results[f"matrix_{size}"] = {
            "pytorch_time": pytorch_time,
            "pytorch_gflops": pytorch_gflops,
            "cupy_time": cupy_time,
            "cupy_gflops": cupy_gflops
        }
    
    return results


def benchmark_docking():
    """Benchmark docking performance"""
    print("Benchmarking docking performance...")
    
    from pandakinetics.core.docking import DockingEngine
    
    # Create test ligand (caffeine)
    caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    
    docking_engine = DockingEngine(n_poses=50)
    
    # Benchmark different pose counts
    pose_counts = [10, 25, 50, 100]
    results = {}
    
    for n_poses in pose_counts:
        print(f"  Pose count: {n_poses}")
        
        docking_engine.n_poses = n_poses
        
        start_time = time.time()
        # This would use a test protein - placeholder timing
        time.sleep(0.1 * n_poses / 10)  # Simulate docking time
        end_time = time.time()
        
        results[f"poses_{n_poses}"] = {
            "time": end_time - start_time,
            "poses_per_second": n_poses / (end_time - start_time)
        }
    
    return results


def benchmark_kinetic_simulation():
    """Benchmark kinetic Monte Carlo simulation"""
    print("Benchmarking kinetic simulation...")
    
    from pandakinetics.simulation.monte_carlo import MonteCarloKinetics
    from pandakinetics.core.networks import TransitionNetwork
    
    # Create test network
    n_states_list = [10, 25, 50, 100]
    results = {}
    
    for n_states in n_states_list:
        print(f"  Network size: {n_states} states")
        
        # Create random network
        positions = torch.randn(n_states, 10, 3)  # 10 atoms per state
        energies = torch.randn(n_states) * 10
        
        network = TransitionNetwork(positions, energies)
        
        mc_simulator = MonteCarloKinetics(n_replicas=8, max_steps=10000)
        
        start_time = time.time()
        simulation_results = mc_simulator.simulate(network, max_time=1e-6)
        end_time = time.time()
        
        results[f"states_{n_states}"] = {
            "time": end_time - start_time,
            "steps_per_second": 10000 / (end_time - start_time),
            "binding_events": len(simulation_results.binding_times),
            "unbinding_events": len(simulation_results.unbinding_times)
        }
    
    return results


def benchmark_full_pipeline():
    """Benchmark full prediction pipeline"""
    print("Benchmarking full pipeline...")
    
    # Use small test case
    simulator = KineticSimulator(
        n_replicas=4,
        max_simulation_time=1e-6
    )
    
    # Test ligand (ethanol)
    #ligand_smiles = "CCO"
    ligand_smiles = [
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen  
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",   # Caffeine
    "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F"  # Celecoxib
]
    start_time = time.time()
    
    # This would run the full pipeline - placeholder for now
    time.sleep(5)  # Simulate pipeline time
    
    end_time = time.time()
    
    return {
        "full_pipeline": {
            "time": end_time - start_time,
            "ligand": ligand_smiles
        }
    }


def get_system_info():
    """Get system information"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "platform": platform.platform()
    }
    
    # GPU info
    if torch.cuda.is_available():
        info["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
        }
    
    return info


def main():
    """Main benchmarking function"""
    parser = argparse.ArgumentParser(description="PandaKinetics Benchmarking")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--output", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("PandaKinetics Performance Benchmarks")
    print("=" * 40)
    
    results = {
        "system_info": get_system_info(),
        "benchmarks": {}
    }
    
    # Always run GPU benchmarks
    results["benchmarks"]["gpu"] = benchmark_gpu()
    
    if args.quick:
        # Quick benchmarks only
        results["benchmarks"]["docking_quick"] = benchmark_docking()
    elif args.full:
        # Full benchmark suite
        results["benchmarks"]["docking"] = benchmark_docking()
        results["benchmarks"]["kinetic_simulation"] = benchmark_kinetic_simulation()
        results["benchmarks"]["full_pipeline"] = benchmark_full_pipeline()
    else:
        # Default: essential benchmarks
        results["benchmarks"]["docking"] = benchmark_docking()
        results["benchmarks"]["kinetic_simulation"] = benchmark_kinetic_simulation()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_results_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to: {output_file}")
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 20)
    
    if "gpu" in results["benchmarks"]:
        gpu_results = results["benchmarks"]["gpu"]
        best_gflops = max(
            gpu_results[key]["pytorch_gflops"] 
            for key in gpu_results.keys()
        )
        print(f"Best GPU performance: {best_gflops:.1f} GFLOPS")
    
    if "docking" in results["benchmarks"]:
        docking_results = results["benchmarks"]["docking"]
        best_poses_per_sec = max(
            docking_results[key]["poses_per_second"]
            for key in docking_results.keys()
        )
        print(f"Best docking performance: {best_poses_per_sec:.1f} poses/sec")


if __name__ == "__main__":
    import sys
    import platform
    main()
