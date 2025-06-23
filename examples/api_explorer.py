#!/usr/bin/env python3
"""
PandaKinetics API Explorer
Run this first to understand the actual API structure
"""

import inspect
from pandakinetics import KineticSimulator
from pandakinetics.core.docking import DockingEngine
from pandakinetics.simulation.monte_carlo import MonteCarloKinetics
from pandakinetics.core.networks import TransitionNetwork

def explore_class_api(cls, class_name):
    """Explore a class's methods and attributes"""
    
    print(f"\n=== {class_name} API ===")
    
    # Get all methods and attributes
    members = inspect.getmembers(cls)
    
    # Separate methods and attributes
    methods = []
    attributes = []
    
    for name, obj in members:
        if name.startswith('_'):
            continue
            
        if inspect.ismethod(obj) or inspect.isfunction(obj):
            # Get method signature
            try:
                sig = inspect.signature(obj)
                methods.append(f"{name}{sig}")
            except:
                methods.append(f"{name}(...)")
        else:
            attributes.append(f"{name}: {type(obj).__name__}")
    
    print(f"Methods ({len(methods)}):")
    for method in sorted(methods):
        print(f"  {method}")
    
    print(f"\nAttributes ({len(attributes)}):")
    for attr in sorted(attributes):
        print(f"  {attr}")
    
    return methods, attributes

def explore_instance_api(instance, instance_name):
    """Explore an instance's available methods"""
    
    print(f"\n=== {instance_name} Instance API ===")
    
    # Get all callable methods
    methods = []
    for name in dir(instance):
        if name.startswith('_'):
            continue
        
        attr = getattr(instance, name)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
                methods.append(f"{name}{sig}")
            except:
                methods.append(f"{name}(...)")
    
    print(f"Available methods ({len(methods)}):")
    for method in sorted(methods):
        print(f"  {method}")
    
    return methods

def main():
    """Explore all PandaKinetics APIs"""
    
    print("PandaKinetics API Explorer")
    print("=" * 40)
    
    # 1. Explore KineticSimulator class
    try:
        explore_class_api(KineticSimulator, "KineticSimulator")
        
        # Create instance and explore
        simulator = KineticSimulator(n_replicas=4, max_simulation_time=1e-6)
        explore_instance_api(simulator, "KineticSimulator")
        
    except Exception as e:
        print(f"Error exploring KineticSimulator: {e}")
    
    # 2. Explore DockingEngine
    try:
        explore_class_api(DockingEngine, "DockingEngine")
        
        docking_engine = DockingEngine(n_poses=10)
        explore_instance_api(docking_engine, "DockingEngine")
        
    except Exception as e:
        print(f"Error exploring DockingEngine: {e}")
    
    # 3. Explore MonteCarloKinetics
    try:
        explore_class_api(MonteCarloKinetics, "MonteCarloKinetics")
        
        mc_sim = MonteCarloKinetics(n_replicas=4, max_steps=1000)
        explore_instance_api(mc_sim, "MonteCarloKinetics")
        
    except Exception as e:
        print(f"Error exploring MonteCarloKinetics: {e}")
    
    # 4. Explore TransitionNetwork
    try:
        explore_class_api(TransitionNetwork, "TransitionNetwork")
        
        # Try to create a simple network
        import torch
        positions = torch.randn(5, 10, 3)  # 5 states, 10 atoms, 3D coords
        energies = torch.randn(5)
        
        network = TransitionNetwork(positions, energies)
        explore_instance_api(network, "TransitionNetwork")
        
    except Exception as e:
        print(f"Error exploring TransitionNetwork: {e}")
    
    # 5. Check for common patterns
    print(f"\n=== Common API Patterns ===")
    
    # Check if there are any obvious workflow methods
    simulator = KineticSimulator(n_replicas=4, max_simulation_time=1e-6)
    
    # Common method names to check
    common_methods = [
        'run', 'simulate', 'analyze', 'predict', 'dock', 'score',
        'calculate', 'compute', 'process', 'execute', 'perform',
        'bind', 'unbind', 'kinetics', 'dynamics', 'trajectory'
    ]
    
    found_methods = []
    for method_name in common_methods:
        if hasattr(simulator, method_name):
            method = getattr(simulator, method_name)
            if callable(method):
                try:
                    sig = inspect.signature(method)
                    found_methods.append(f"simulator.{method_name}{sig}")
                except:
                    found_methods.append(f"simulator.{method_name}(...)")
    
    if found_methods:
        print("Potential workflow methods found:")
        for method in found_methods:
            print(f"  {method}")
    else:
        print("No common workflow methods found in KineticSimulator")

if __name__ == "__main__":
    main()
