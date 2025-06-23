# =============================================================================
# pandakinetics/simulation/analysis.py - Complete Implementation
# =============================================================================

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import pandas as pd

# Import shared types
from ..types import SimulationResults

# Import utilities with fallback
try:
    from ..utils.gpu_utils import GPUUtils
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    
    class GPUUtils:
        @staticmethod
        def get_device(device=None):
            return torch.device('cpu')


class TrajectoryAnalyzer:
    """
    Analysis tools for kinetic simulation trajectories
    
    Provides methods to analyze kinetic Monte Carlo trajectories,
    calculate statistical properties, and generate visualizations.
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize trajectory analyzer"""
        if GPU_UTILS_AVAILABLE:
            self.device = GPUUtils.get_device(device)
        else:
            self.device = torch.device('cpu')
        logger.info(f"TrajectoryAnalyzer initialized on {self.device}")
    
    def analyze_kinetics(self, results: SimulationResults) -> Dict[str, float]:
        """
        Comprehensive kinetic analysis
        
        Args:
            results: Simulation results from kinetic Monte Carlo
            
        Returns:
            Dictionary with kinetic parameters and statistics
        """
        logger.info("Performing kinetic analysis")
        
        analysis = {}
        
        # Basic statistics
        analysis.update(self._calculate_basic_stats(results))
        
        # Rate analysis
        analysis.update(self._analyze_rates(results))
        
        # Pathway analysis
        analysis.update(self._analyze_pathways(results))
        
        # Temporal analysis
        analysis.update(self._analyze_temporal_behavior(results))
        
        return analysis
    
    def _calculate_basic_stats(self, results: SimulationResults) -> Dict[str, float]:
        """Calculate basic statistical properties"""
        
        stats = {}
        
        # Binding events
        if len(results.binding_times) > 0:
            stats['mean_binding_time'] = torch.mean(results.binding_times).item()
            stats['std_binding_time'] = torch.std(results.binding_times).item()
            stats['median_binding_time'] = torch.median(results.binding_times).item()
            stats['n_binding_events'] = len(results.binding_times)
        else:
            stats.update({
                'mean_binding_time': 0.0,
                'std_binding_time': 0.0,
                'median_binding_time': 0.0,
                'n_binding_events': 0
            })
        
        # Unbinding events
        if len(results.unbinding_times) > 0:
            stats['mean_unbinding_time'] = torch.mean(results.unbinding_times).item()
            stats['std_unbinding_time'] = torch.std(results.unbinding_times).item()
            stats['median_unbinding_time'] = torch.median(results.unbinding_times).item()
            stats['n_unbinding_events'] = len(results.unbinding_times)
        else:
            stats.update({
                'mean_unbinding_time': 0.0,
                'std_unbinding_time': 0.0,
                'median_unbinding_time': 0.0,
                'n_unbinding_events': 0
            })
        
        # State populations
        if len(results.state_populations) > 0:
            stats['max_population'] = torch.max(results.state_populations).item()
            stats['min_population'] = torch.min(results.state_populations).item()
            stats['population_entropy'] = self._calculate_entropy(results.state_populations)
        else:
            stats.update({
                'max_population': 0.0,
                'min_population': 0.0,
                'population_entropy': 0.0
            })
        
        return stats
    
    def _analyze_rates(self, results: SimulationResults) -> Dict[str, float]:
        """Analyze kinetic rates"""
        
        rates = {}
        
        # Association rate
        if len(results.binding_times) > 0:
            # Assume 1 μM concentration
            concentration = 1e-6  # M
            mean_binding_time = torch.mean(results.binding_times).item()
            kon = 1.0 / (mean_binding_time * concentration)  # M^-1 s^-1
            rates['kon'] = kon
        else:
            rates['kon'] = 0.0
        
        # Dissociation rate
        if len(results.unbinding_times) > 0:
            mean_residence_time = torch.mean(results.unbinding_times).item()
            koff = 1.0 / mean_residence_time  # s^-1
            rates['koff'] = koff
        else:
            rates['koff'] = 0.0
        
        # Derived quantities
        if rates['kon'] > 0:
            rates['kd'] = rates['koff'] / rates['kon']  # M
        else:
            rates['kd'] = float('inf')
        
        if rates['koff'] > 0:
            rates['residence_time'] = 1.0 / rates['koff']  # s
        else:
            rates['residence_time'] = float('inf')
        
        return rates
    
    def _analyze_pathways(self, results: SimulationResults) -> Dict[str, float]:
        """Analyze dominant pathways"""
        
        pathway_stats = {}
        
        # Count transitions
        total_transitions = torch.sum(results.transition_counts).item()
        
        if total_transitions > 0:
            # Most frequent transitions
            max_transitions = torch.max(results.transition_counts).item()
            pathway_stats['max_transition_frequency'] = max_transitions / total_transitions
            
            # Number of active pathways
            active_transitions = torch.sum(results.transition_counts > 0).item()
            pathway_stats['n_active_transitions'] = active_transitions
            
            # Pathway diversity (entropy)
            transition_probs = results.transition_counts / total_transitions
            pathway_stats['pathway_entropy'] = self._calculate_entropy(transition_probs)
        else:
            pathway_stats.update({
                'max_transition_frequency': 0.0,
                'n_active_transitions': 0,
                'pathway_entropy': 0.0
            })
        
        return pathway_stats
    
    def _analyze_temporal_behavior(self, results: SimulationResults) -> Dict[str, float]:
        """Analyze temporal behavior of trajectories"""
        
        temporal_stats = {}
        
        # Trajectory lengths
        traj_lengths = [len(traj) for traj in results.trajectories if len(traj) > 0]
        
        if traj_lengths:
            temporal_stats['mean_trajectory_length'] = np.mean(traj_lengths)
            temporal_stats['std_trajectory_length'] = np.std(traj_lengths)
            temporal_stats['max_trajectory_length'] = np.max(traj_lengths)
            temporal_stats['min_trajectory_length'] = np.min(traj_lengths)
        else:
            temporal_stats.update({
                'mean_trajectory_length': 0.0,
                'std_trajectory_length': 0.0,
                'max_trajectory_length': 0.0,
                'min_trajectory_length': 0.0
            })
        
        # Time correlations
        if len(results.binding_times) > 1:
            binding_autocorr = self._calculate_autocorrelation(results.binding_times)
            temporal_stats['binding_autocorrelation'] = binding_autocorr
        else:
            temporal_stats['binding_autocorrelation'] = 0.0
        
        if len(results.unbinding_times) > 1:
            unbinding_autocorr = self._calculate_autocorrelation(results.unbinding_times)
            temporal_stats['unbinding_autocorrelation'] = unbinding_autocorr
        else:
            temporal_stats['unbinding_autocorrelation'] = 0.0
        
        return temporal_stats
    
    def _calculate_entropy(self, probabilities: torch.Tensor) -> float:
        """Calculate Shannon entropy"""
        
        # Handle empty tensors
        if len(probabilities) == 0:
            return 0.0
        
        # Normalize probabilities
        prob_sum = torch.sum(probabilities)
        if prob_sum == 0:
            return 0.0
        
        probs = probabilities / prob_sum
        
        # Remove zeros to avoid log(0)
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 0.0
        
        entropy = -torch.sum(probs * torch.log(probs)).item()
        return entropy
    
    def _calculate_autocorrelation(self, times: torch.Tensor, max_lag: int = 10) -> float:
        """Calculate autocorrelation function"""
        
        if len(times) < 2:
            return 0.0
        
        # Convert to numpy for easier manipulation
        times_np = times.cpu().numpy()
        
        # Calculate time intervals
        intervals = np.diff(times_np)
        
        if len(intervals) < 2:
            return 0.0
        
        if len(intervals) < max_lag:
            max_lag = len(intervals) - 1
        
        if max_lag <= 0:
            return 0.0
        
        # Calculate autocorrelation at lag 1
        if len(intervals) > 1:
            try:
                autocorr = np.corrcoef(intervals[:-1], intervals[1:])[0, 1]
                return float(autocorr) if not np.isnan(autocorr) else 0.0
            except:
                return 0.0
        else:
            return 0.0
    
    def plot_kinetic_summary(self, results: SimulationResults, save_path: Optional[str] = None):
        """Create comprehensive kinetic summary plots"""
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Kinetic Simulation Summary', fontsize=16)
            
            # Plot 1: Binding/unbinding times histogram
            if len(results.binding_times) > 0:
                axes[0, 0].hist(results.binding_times.cpu().numpy(), bins=20, alpha=0.7, 
                               label='Binding', color='blue')
            if len(results.unbinding_times) > 0:
                axes[0, 0].hist(results.unbinding_times.cpu().numpy(), bins=20, alpha=0.7, 
                               label='Unbinding', color='red')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Event Time Distributions')
            axes[0, 0].legend()
            
            # Plot 2: State populations
            if len(results.state_populations) > 0:
                populations = results.state_populations.cpu().numpy()
                state_indices = np.arange(len(populations))
                axes[0, 1].bar(state_indices, populations)
                axes[0, 1].set_xlabel('State Index')
                axes[0, 1].set_ylabel('Population')
                axes[0, 1].set_title('State Populations')
            
            # Plot 3: Transition matrix heatmap
            if results.transition_counts.numel() > 0:
                transition_matrix = results.transition_counts.cpu().numpy()
                im = axes[0, 2].imshow(transition_matrix, cmap='viridis', aspect='auto')
                axes[0, 2].set_xlabel('To State')
                axes[0, 2].set_ylabel('From State')
                axes[0, 2].set_title('Transition Count Matrix')
                plt.colorbar(im, ax=axes[0, 2])
            
            # Plot 4: Sample trajectory
            if results.trajectories and len(results.trajectories[0]) > 0:
                traj = results.trajectories[0].cpu().numpy()
                if len(traj) > 0 and traj.shape[1] >= 2:
                    axes[1, 0].plot(traj[:, 0], traj[:, 1])
                    axes[1, 0].set_xlabel('Time (s)')
                    axes[1, 0].set_ylabel('State')
                    axes[1, 0].set_title('Sample Trajectory')
            
            # Plot 5: Cumulative events
            if len(results.binding_times) > 0:
                binding_sorted = torch.sort(results.binding_times)[0].cpu().numpy()
                axes[1, 1].plot(binding_sorted, np.arange(1, len(binding_sorted)+1), 
                               label='Binding', color='blue')
            if len(results.unbinding_times) > 0:
                unbinding_sorted = torch.sort(results.unbinding_times)[0].cpu().numpy()
                axes[1, 1].plot(unbinding_sorted, np.arange(1, len(unbinding_sorted)+1), 
                               label='Unbinding', color='red')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Cumulative Events')
            axes[1, 1].set_title('Cumulative Event Count')
            axes[1, 1].legend()
            
            # Plot 6: Rate estimates over time
            if len(results.binding_times) > 5 and len(results.unbinding_times) > 5:
                # Calculate rolling estimates
                n_points = min(20, len(results.binding_times)//2)
                time_points = np.linspace(0.1, 0.9, n_points)
                kon_estimates = []
                koff_estimates = []
                
                for frac in time_points:
                    n_bind = int(frac * len(results.binding_times))
                    n_unbind = int(frac * len(results.unbinding_times))
                    
                    if n_bind > 0:
                        mean_bind_time = torch.mean(results.binding_times[:n_bind]).item()
                        kon_est = 1.0 / (mean_bind_time * 1e-6)  # Assume 1 μM
                        kon_estimates.append(kon_est)
                    else:
                        kon_estimates.append(0)
                    
                    if n_unbind > 0:
                        mean_unbind_time = torch.mean(results.unbinding_times[:n_unbind]).item()
                        koff_est = 1.0 / mean_unbind_time
                        koff_estimates.append(koff_est)
                    else:
                        koff_estimates.append(0)
                
                axes[1, 2].plot(time_points, kon_estimates, 'b-', label='kon')
                axes[1, 2].plot(time_points, koff_estimates, 'r-', label='koff')
                axes[1, 2].set_xlabel('Fraction of Data')
                axes[1, 2].set_ylabel('Rate Estimate')
                axes[1, 2].set_title('Rate Convergence')
                axes[1, 2].legend()
                axes[1, 2].set_yscale('log')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Kinetic summary plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create plots: {e}")
            logger.info("Plotting requires matplotlib and may need display backend")
    
    def plot_energy_landscape(
        self, 
        results: SimulationResults, 
        network=None,  # TransitionNetwork
        save_path: Optional[str] = None
    ):
        """Plot energy landscape and transition pathways"""
        
        try:
            if network is None:
                logger.warning("Network required for energy landscape plot")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Energy landscape
            if hasattr(network, 'energies') and len(network.energies) > 0:
                energies = network.energies.cpu().numpy()
                populations = results.state_populations.cpu().numpy()
                
                scatter = ax1.scatter(
                    range(len(energies)), 
                    energies, 
                    s=populations * 1000,  # Size by population
                    c=populations,
                    cmap='viridis',
                    alpha=0.7
                )
                ax1.set_xlabel('State Index')
                ax1.set_ylabel('Energy (kcal/mol)')
                ax1.set_title('Energy Landscape')
                plt.colorbar(scatter, ax=ax1, label='Population')
            
            # Plot 2: Transition network
            if results.transition_counts.numel() > 0:
                transition_matrix = results.transition_counts.cpu().numpy()
                
                # Create network graph
                import networkx as nx
                G = nx.Graph()
                
                n_states = transition_matrix.shape[0]
                for i in range(n_states):
                    for j in range(i+1, n_states):
                        if transition_matrix[i, j] > 0:
                            G.add_edge(i, j, weight=transition_matrix[i, j])
                
                if len(G.nodes()) > 0:
                    pos = nx.spring_layout(G)
                    
                    # Draw nodes
                    node_sizes = [results.state_populations[i].item() * 1000 for i in G.nodes()]
                    nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=node_sizes, 
                                         node_color='lightblue', alpha=0.7)
                    
                    # Draw edges
                    edges = G.edges()
                    weights = [G[u][v]['weight'] for u, v in edges]
                    nx.draw_networkx_edges(G, pos, ax=ax2, width=weights, alpha=0.5)
                    
                    # Draw labels
                    nx.draw_networkx_labels(G, pos, ax=ax2)
                    
                    ax2.set_title('Transition Network')
                    ax2.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Energy landscape plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to create energy landscape plot: {e}")
    
    def generate_report(self, results: SimulationResults, analysis: Dict[str, float]) -> str:
        """Generate comprehensive analysis report"""
        
        report = """
PandaKinetics Simulation Report
================================

SIMULATION SUMMARY
------------------
Total simulation time: {:.2e} s
Number of binding events: {}
Number of unbinding events: {}
Number of active transitions: {}

KINETIC PARAMETERS
------------------
Association rate (kon): {:.2e} M⁻¹s⁻¹
Dissociation rate (koff): {:.2e} s⁻¹
Binding affinity (Kd): {:.2e} M
Residence time: {:.2e} s

STATISTICAL ANALYSIS
--------------------
Mean binding time: {:.2e} ± {:.2e} s
Mean unbinding time: {:.2e} ± {:.2e} s
State population entropy: {:.3f}
Pathway entropy: {:.3f}

TEMPORAL BEHAVIOR
-----------------
Mean trajectory length: {:.1f} steps
Binding autocorrelation: {:.3f}
Unbinding autocorrelation: {:.3f}

CONFIDENCE ASSESSMENT
--------------------
Binding events sufficient: {}
Unbinding events sufficient: {}
Simulation converged: {}
        """.format(
            results.total_simulation_time,
            analysis.get('n_binding_events', 0),
            analysis.get('n_unbinding_events', 0),
            analysis.get('n_active_transitions', 0),
            analysis.get('kon', 0),
            analysis.get('koff', 0),
            analysis.get('kd', float('inf')),
            analysis.get('residence_time', float('inf')),
            analysis.get('mean_binding_time', 0),
            analysis.get('std_binding_time', 0),
            analysis.get('mean_unbinding_time', 0),
            analysis.get('std_unbinding_time', 0),
            analysis.get('population_entropy', 0),
            analysis.get('pathway_entropy', 0),
            analysis.get('mean_trajectory_length', 0),
            analysis.get('binding_autocorrelation', 0),
            analysis.get('unbinding_autocorrelation', 0),
            "Yes" if analysis.get('n_binding_events', 0) >= 10 else "No",
            "Yes" if analysis.get('n_unbinding_events', 0) >= 10 else "No",
            "Yes" if analysis.get('n_binding_events', 0) >= 20 and 
                     analysis.get('n_unbinding_events', 0) >= 20 else "No"
        )
        
        return report
    
    def export_results(self, results: SimulationResults, analysis: Dict[str, float], filename: str):
        """Export results to various formats"""
        
        try:
            # Prepare data for export
            export_data = {
                'simulation_summary': {
                    'total_time': results.total_simulation_time,
                    'n_binding_events': len(results.binding_times),
                    'n_unbinding_events': len(results.unbinding_times),
                },
                'kinetic_parameters': {
                    'kon': analysis.get('kon', 0),
                    'koff': analysis.get('koff', 0),
                    'kd': analysis.get('kd', float('inf')),
                    'residence_time': analysis.get('residence_time', float('inf'))
                },
                'statistics': analysis
            }
            
            # Export to JSON
            import json
            with open(f"{filename}.json", 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            # Export to CSV (tabular data)
            df_data = []
            for key, value in analysis.items():
                df_data.append({'parameter': key, 'value': value})
            
            df = pd.DataFrame(df_data)
            df.to_csv(f"{filename}.csv", index=False)
            
            logger.info(f"Results exported to {filename}.json and {filename}.csv")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")


# Utility functions for analysis
def calculate_mean_first_passage_time(
    transition_matrix: torch.Tensor, 
    source_states: List[int], 
    target_states: List[int]
) -> float:
    """Calculate mean first passage time between state sets"""
    
    try:
        # Convert to numpy for linear algebra
        T = transition_matrix.cpu().numpy()
        n_states = T.shape[0]
        
        # Set up absorbing boundary conditions
        T_modified = T.copy()
        for target in target_states:
            T_modified[target, :] = 0
            T_modified[target, target] = 1
        
        # Solve for mean first passage times
        # (I - T)t = 1, where t is the MFPT vector
        I = np.eye(n_states)
        A = I - T_modified
        
        # Remove target states from the system
        active_states = [i for i in range(n_states) if i not in target_states]
        A_reduced = A[np.ix_(active_states, active_states)]
        b = np.ones(len(active_states))
        
        # Solve linear system
        mfpt = np.linalg.solve(A_reduced, b)
        
        # Average over source states
        source_indices = [active_states.index(s) for s in source_states if s in active_states]
        if source_indices:
            return float(np.mean(mfpt[source_indices]))
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"MFPT calculation failed: {e}")
        return 0.0


def calculate_committor_probabilities(
    transition_matrix: torch.Tensor,
    source_states: List[int],
    target_states: List[int]
) -> torch.Tensor:
    """Calculate committor probabilities for each state"""
    
    try:
        T = transition_matrix.cpu().numpy()
        n_states = T.shape[0]
        
        # Set up boundary conditions
        # Committor = 0 for source states, 1 for target states
        committor = np.zeros(n_states)
        
        # Boundary conditions
        for source in source_states:
            committor[source] = 0.0
        for target in target_states:
            committor[target] = 1.0
        
        # Solve for interior states
        interior_states = [i for i in range(n_states) 
                          if i not in source_states and i not in target_states]
        
        if interior_states:
            # (I - T)q = 0 for interior states with boundary conditions
            A = np.eye(len(interior_states)) - T[np.ix_(interior_states, interior_states)]
            b = np.zeros(len(interior_states))
            
            # Add boundary contributions
            for i, state in enumerate(interior_states):
                for target in target_states:
                    b[i] += T[state, target]
            
            # Solve
            q_interior = np.linalg.solve(A, b)
            
            # Fill in results
            for i, state in enumerate(interior_states):
                committor[state] = q_interior[i]
        
        return torch.tensor(committor, dtype=torch.float32)
        
    except Exception as e:
        logger.error(f"Committor calculation failed: {e}")
        return torch.zeros(transition_matrix.shape[0])


# Export main classes
__all__ = [
    'TrajectoryAnalyzer',
    'calculate_mean_first_passage_time',
    'calculate_committor_probabilities'
]