# =============================================================================
# pandakinetics/utils/gpu_utils.py
# =============================================================================

import torch
import cupy as cp
import numpy as np
from typing import Optional, Union
from loguru import logger
import psutil
import subprocess


class GPUUtils:
    """Utilities for GPU management and optimization"""
    
    @staticmethod
    def get_device(device: Optional[str] = None) -> torch.device:
        """Get optimal GPU device"""
        
        if device is None:
            if torch.cuda.is_available():
                # Auto-select GPU with most free memory
                device = GPUUtils.select_best_gpu()
            else:
                raise RuntimeError("CUDA not available. PandaKinetics requires GPU.")
        
        if isinstance(device, str):
            device = torch.device(device)
        
        logger.info(f"Using device: {device}")
        return device
    
    @staticmethod
    def select_best_gpu() -> str:
        """Select GPU with most available memory"""
        
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPUs available")
        
        best_gpu = 0
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                total_memory = torch.cuda.get_device_properties(i).total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                free_memory = total_memory - allocated_memory
                
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu = i
        
        logger.info(f"Selected GPU {best_gpu} with {max_free_memory/1e9:.1f} GB free memory")
        return f"cuda:{best_gpu}"
    
    @staticmethod
    def optimize_memory():
        """Optimize GPU memory usage"""
        
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            logger.info("GPU memory optimized")
    
    @staticmethod
    def get_memory_info() -> dict:
        """Get GPU memory information"""
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        info = {}
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i)
                cached = torch.cuda.memory_reserved(i)
                
                info[f"gpu_{i}"] = {
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "allocated_memory": allocated,
                    "cached_memory": cached,
                    "free_memory": props.total_memory - allocated
                }
        
        return info
    
    @staticmethod
    def benchmark_gpu() -> dict:
        """Benchmark GPU performance"""
        
        device = GPUUtils.get_device()
        
        # Matrix multiplication benchmark
        size = 4096
        n_iterations = 10
        
        # Warm up
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(n_iterations):
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        gflops = (2 * size**3 * n_iterations) / (total_time * 1e9)
        
        return {
            "device": str(device),
            "matrix_size": size,
            "iterations": n_iterations,
            "total_time": total_time,
            "gflops": gflops
        }


def check_gpu_availability() -> bool:
    """Check if GPU is available and properly configured"""
    
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.error("CUDA not available")
            return False
        
        # Check CuPy availability
        try:
            import cupy as cp
            cp.cuda.Device(0).use()
            test_array = cp.array([1, 2, 3])
            logger.info("CuPy available and working")
        except Exception as e:
            logger.error(f"CuPy not working: {e}")
            return False
        
        # Check GPU memory
        gpu_info = GPUUtils.get_memory_info()
        for gpu_id, info in gpu_info.items():
            if info["free_memory"] < 2e9:  # Less than 2GB free
                logger.warning(f"{gpu_id}: Low memory ({info['free_memory']/1e9:.1f} GB free)")
        
        logger.info("GPU availability check passed")
        return True
        
    except Exception as e:
        logger.error(f"GPU availability check failed: {e}")
        return False
