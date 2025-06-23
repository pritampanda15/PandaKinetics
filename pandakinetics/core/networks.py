# =============================================================================
# pandakinetics/core/networks.py
# =============================================================================

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from loguru import logger

from ..utils.gpu_utils import GPUUtils


class TransitionNetwork:
    """
    Represents the transition network for ligand binding states
    
    Each node represents a binding state (pose), edges represent transitions
    with associated energy barriers and transition rates.
    """
    
    def __init__(
        self,
        positions: torch.Tensor,
        energies: torch.Tensor,
        temperature: float = 310.0,
        device: Optional[str] = None,
        connectivity_threshold: float = 2.0  # Angstroms
    ):
        """
        Initialize transition network
        
        Args:
            positions: Tensor of shape (n_states, n_atoms, 3) - state coordinates
            energies: Tensor of shape (n_states,) - state energies
            temperature: Temperature in Kelvin
            device: GPU device
            connectivity_threshold: Distance threshold for state connectivity
        """
        self.device = GPUUtils.get_device(device)
        self.temperature = temperature
        self.kT = 8.314e-3 * temperature  # kJ/mol
        
        # Move data to GPU
        self.positions = positions.to(self.device)
        self.energies = energies.to(self.device)
        self.n_states = len(positions)
        
        # Initialize network components
        self.adjacency_matrix = self._build_adjacency_matrix(connectivity_threshold)
        self.barriers = torch.zeros((self.n_states, self.n_states), device=self.device)
        self.rates = torch.zeros((self.n_states, self.n_states), device=self.device)
        
        # Build NetworkX graph for analysis
        self.graph = self._build_networkx_graph()
        
        logger.info(f"TransitionNetwork initialized with {self.n_states} states")
    
    def _build_adjacency_matrix(self, threshold: float) -> torch.Tensor:
        """Build adjacency matrix based on structural similarity"""
        
        # Calculate pairwise distances between states
        distances = torch.cdist(
            self.positions.view(self.n_states, -1),
            self.positions.view(self.n_states, -1)
        )
        
        # Connect states within threshold distance
        adjacency = (distances < threshold).float()
        
        # Remove self-connections
        adjacency.fill_diagonal_(0)
        
        # Ensure symmetry
        adjacency = (adjacency + adjacency.T) / 2
        adjacency = (adjacency > 0).float()
        
        return adjacency
    
    def _build_networkx_graph(self) -> nx.Graph:
        """Build NetworkX graph for network analysis"""
        G = nx.Graph()
        
        # Add nodes with attributes
        for i in range(self.n_states):
            G.add_node(i, energy=self.energies[i].item())
        
        # Add edges
        adj_cpu = self.adjacency_matrix.cpu().numpy()
        for i in range(self.n_states):
            for j in range(i+1, self.n_states):
                if adj_cpu[i, j] > 0:
                    G.add_edge(i, j)
        
        return G
    
    def update_barriers(self, barriers: torch.Tensor):
        """Update energy barriers and recalculate transition rates"""
        self.barriers = barriers.to(self.device)
        self._calculate_transition_rates()
    
    def _calculate_transition_rates(self):
        """Calculate transition rates using transition state theory"""
        
        # Eyring equation: k = (kT/h) * exp(-ΔG‡/RT)
        h = 6.626e-34  # Planck constant (J⋅s)
        k_B = 1.381e-23  # Boltzmann constant (J/K)
        
        prefactor = k_B * self.temperature / h  # s⁻¹
        
        # Calculate rates only for connected states
        mask = self.adjacency_matrix > 0
        
        with torch.no_grad():
            self.rates = torch.zeros_like(self.barriers)
            
            # Apply Eyring equation where connected
            valid_barriers = torch.where(mask, self.barriers, torch.inf)
            exp_term = torch.exp(-valid_barriers / self.kT)
            self.rates = torch.where(mask, prefactor * exp_term, 0.0)
    
    def get_rate_matrix(self) -> torch.Tensor:
        """Get the rate matrix for kinetic Monte Carlo"""
        return self.rates
    
    def get_equilibrium_populations(self) -> torch.Tensor:
        """Calculate equilibrium populations using detailed balance"""
        
        # Boltzmann distribution
        relative_energies = self.energies - torch.min(self.energies)
        populations = torch.exp(-relative_energies / self.kT)
        populations = populations / torch.sum(populations)
        
        return populations
    
    def find_binding_sites(self) -> List[int]:
        """Identify bound states (local energy minima)"""
        
        # Find states with energy below average
        mean_energy = torch.mean(self.energies)
        binding_sites = torch.where(self.energies < mean_energy)[0].tolist()
        
        return binding_sites
    
    def find_unbound_states(self) -> List[int]:
        """Identify unbound states (high energy states)"""
        
        # Find states with energy above threshold
        energy_threshold = torch.mean(self.energies) + torch.std(self.energies)
        unbound_states = torch.where(self.energies > energy_threshold)[0].tolist()
        
        return unbound_states
    
    def find_nearest_state(self, coordinates: torch.Tensor) -> int:
        """Find the nearest state to given coordinates"""
        
        coords_flat = coordinates.view(-1).to(self.device)
        positions_flat = self.positions.view(self.n_states, -1)
        
        distances = torch.norm(positions_flat - coords_flat, dim=1)
        nearest_state = torch.argmin(distances).item()
        
        return nearest_state
    
    def analyze_connectivity(self) -> Dict[str, float]:
        """Analyze network connectivity properties"""
        
        G = self.graph
        
        analysis = {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "density": nx.density(G),
            "n_connected_components": nx.number_connected_components(G),
            "average_clustering": nx.average_clustering(G),
        }
        
        if nx.is_connected(G):
            analysis["diameter"] = nx.diameter(G)
            analysis["average_path_length"] = nx.average_shortest_path_length(G)
        
        return analysis
    
    def find_critical_pathways(self, source_states: List[int], target_states: List[int]) -> List[List[int]]:
        """Find critical pathways between source and target states"""
        
        pathways = []
        
        for source in source_states:
            for target in target_states:
                try:
                    # Find shortest path
                    path = nx.shortest_path(self.graph, source, target)
                    pathways.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return pathways
    
    def calculate_flux(self, populations: torch.Tensor) -> torch.Tensor:
        """Calculate flux between states"""
        
        # Flux = rate * population_from - rate_reverse * population_to
        flux = torch.zeros_like(self.rates)
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                if self.rates[i, j] > 0:
                    forward_flux = self.rates[i, j] * populations[i]
                    backward_flux = self.rates[j, i] * populations[j]
                    flux[i, j] = forward_flux - backward_flux
        
        return flux
    
    def export_network(self, filename: str):
        """Export network to file for visualization"""
        
        # Convert to NetworkX and save
        pos = {}
        for i in range(self.n_states):
            # Use first two coordinates for 2D layout
            pos[i] = (
                self.positions[i, 0, 0].item(),
                self.positions[i, 0, 1].item()
            )
        
        # Add positions as node attributes
        nx.set_node_attributes(self.graph, pos, 'pos')
        
        # Save graph
        nx.write_gml(self.graph, filename)
        logger.info(f"Network exported to {filename}")
