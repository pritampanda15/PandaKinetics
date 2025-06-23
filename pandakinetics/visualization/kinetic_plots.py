# =============================================================================
# pandakinetics/visualization/kinetic_plots.py - Kinetic Data Visualization
# =============================================================================

"""
Kinetic data visualization tools for PandaKinetics

Provides plotting functions for kinetic parameters, time series data,
and statistical analysis of simulation results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

# Import shared types with fallback
try:
    from ..types import KineticResults, SimulationResults
except ImportError:
    # Create dummy classes
    class KineticResults:
        def __init__(self):
            pass
    
    class SimulationResults:
        def __init__(self):
            pass


class KineticPlotter:
    """
    Kinetic data visualization and plotting tools
    
    Provides methods for visualizing kinetic parameters, time series,
    and statistical analysis of molecular dynamics simulations.
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize kinetic plotter
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        
        # Set plotting style
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default")
        
        logger.info(f"KineticPlotter initialized with style '{style}'")
    
    def plot_kinetic_parameters(
        self,
        results: Dict[str, float],
        save_path: Optional[str] = None,
        log_scale: bool = True
    ) -> plt.Figure:
        """
        Plot kinetic parameters (kon, koff, Kd, residence time)
        
        Args:
            results: Dictionary with kinetic parameters
            save_path: Path to save figure
            log_scale: Use log scale for y-axis
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating kinetic parameters plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Kinetic Parameters', fontsize=16)
        
        try:
            # Extract parameters with defaults
            kon = results.get('kon', 0)
            koff = results.get('koff', 0) 
            kd = results.get('kd', results.get('binding_affinity', 0))
            residence_time = results.get('residence_time', 0)
            
            # Plot 1: Association rate (kon)
            if kon > 0:
                axes[0, 0].bar(['kon'], [kon], color='blue', alpha=0.7)
                axes[0, 0].set_ylabel('kon (M⁻¹s⁻¹)')
                axes[0, 0].set_title('Association Rate')
                if log_scale:
                    axes[0, 0].set_yscale('log')
                axes[0, 0].text(0, kon/2, f'{kon:.2e}', ha='center', va='center')
            
            # Plot 2: Dissociation rate (koff)
            if koff > 0:
                axes[0, 1].bar(['koff'], [koff], color='red', alpha=0.7)
                axes[0, 1].set_ylabel('koff (s⁻¹)')
                axes[0, 1].set_title('Dissociation Rate')
                if log_scale:
                    axes[0, 1].set_yscale('log')
                axes[0, 1].text(0, koff/2, f'{koff:.2e}', ha='center', va='center')
            
            # Plot 3: Binding affinity (Kd)
            if kd > 0 and kd != float('inf'):
                axes[1, 0].bar(['Kd'], [kd], color='green', alpha=0.7)
                axes[1, 0].set_ylabel('Kd (M)')
                axes[1, 0].set_title('Binding Affinity')
                if log_scale:
                    axes[1, 0].set_yscale('log')
                axes[1, 0].text(0, kd/2, f'{kd:.2e}', ha='center', va='center')
            
            # Plot 4: Residence time
            if residence_time > 0 and residence_time != float('inf'):
                axes[1, 1].bar(['τ'], [residence_time], color='orange', alpha=0.7)
                axes[1, 1].set_ylabel('Residence Time (s)')
                axes[1, 1].set_title('Residence Time')
                if log_scale:
                    axes[1, 1].set_yscale('log')
                axes[1, 1].text(0, residence_time/2, f'{residence_time:.2e}', ha='center', va='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Kinetic parameters plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create kinetic parameters plot: {e}")
            return fig
    
    def plot_time_series(
        self,
        results: Any,  # SimulationResults
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot time series data from simulation
        
        Args:
            results: SimulationResults object
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating time series plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Simulation Time Series', fontsize=16)
        
        try:
            # Plot 1: Binding events over time
            if hasattr(results, 'binding_times') and len(results.binding_times) > 0:
                binding_times = results.binding_times.cpu().numpy() if hasattr(results.binding_times, 'cpu') else results.binding_times
                axes[0, 0].plot(binding_times, np.arange(1, len(binding_times)+1), 'b-', marker='o')
                axes[0, 0].set_xlabel('Time (s)')
                axes[0, 0].set_ylabel('Cumulative Binding Events')
                axes[0, 0].set_title('Binding Events Over Time')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Unbinding events over time
            if hasattr(results, 'unbinding_times') and len(results.unbinding_times) > 0:
                unbinding_times = results.unbinding_times.cpu().numpy() if hasattr(results.unbinding_times, 'cpu') else results.unbinding_times
                axes[0, 1].plot(unbinding_times, np.arange(1, len(unbinding_times)+1), 'r-', marker='o')
                axes[0, 1].set_xlabel('Time (s)')
                axes[0, 1].set_ylabel('Cumulative Unbinding Events')
                axes[0, 1].set_title('Unbinding Events Over Time')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Sample trajectory
            if hasattr(results, 'trajectories') and results.trajectories and len(results.trajectories[0]) > 0:
                traj = results.trajectories[0]
                if hasattr(traj, 'cpu'):
                    traj = traj.cpu().numpy()
                
                if len(traj) > 0 and traj.shape[1] >= 2:
                    axes[1, 0].plot(traj[:, 0], traj[:, 1], 'g-', alpha=0.7)
                    axes[1, 0].set_xlabel('Time (s)')
                    axes[1, 0].set_ylabel('State')
                    axes[1, 0].set_title('Sample Trajectory')
                    axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Event rate over time (if enough events)
            if (hasattr(results, 'binding_times') and len(results.binding_times) > 10 and
                hasattr(results, 'unbinding_times') and len(results.unbinding_times) > 10):
                
                # Calculate rates in time windows
                max_time = max(
                    np.max(results.binding_times.cpu().numpy() if hasattr(results.binding_times, 'cpu') else results.binding_times),
                    np.max(results.unbinding_times.cpu().numpy() if hasattr(results.unbinding_times, 'cpu') else results.unbinding_times)
                )
                
                time_windows = np.linspace(0, max_time, 20)
                binding_rates = []
                unbinding_rates = []
                
                for i in range(len(time_windows)-1):
                    t_start, t_end = time_windows[i], time_windows[i+1]
                    dt = t_end - t_start
                    
                    binding_times = results.binding_times.cpu().numpy() if hasattr(results.binding_times, 'cpu') else results.binding_times
                    unbinding_times = results.unbinding_times.cpu().numpy() if hasattr(results.unbinding_times, 'cpu') else results.unbinding_times
                    
                    n_binding = np.sum((binding_times >= t_start) & (binding_times < t_end))
                    n_unbinding = np.sum((unbinding_times >= t_start) & (unbinding_times < t_end))
                    
                    binding_rates.append(n_binding / dt if dt > 0 else 0)
                    unbinding_rates.append(n_unbinding / dt if dt > 0 else 0)
                
                window_centers = (time_windows[:-1] + time_windows[1:]) / 2
                axes[1, 1].plot(window_centers, binding_rates, 'b-', label='Binding Rate')
                axes[1, 1].plot(window_centers, unbinding_rates, 'r-', label='Unbinding Rate')
                axes[1, 1].set_xlabel('Time (s)')
                axes[1, 1].set_ylabel('Event Rate (events/s)')
                axes[1, 1].set_title('Event Rates Over Time')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Time series plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create time series plot: {e}")
            return fig
    
    def plot_distribution_analysis(
        self,
        results: Any,  # SimulationResults
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution analysis of binding/unbinding times
        
        Args:
            results: SimulationResults object
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating distribution analysis plot...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Distribution Analysis', fontsize=16)
        
        try:
            # Get data
            binding_times = None
            unbinding_times = None
            
            if hasattr(results, 'binding_times') and len(results.binding_times) > 0:
                binding_times = results.binding_times.cpu().numpy() if hasattr(results.binding_times, 'cpu') else results.binding_times
            
            if hasattr(results, 'unbinding_times') and len(results.unbinding_times) > 0:
                unbinding_times = results.unbinding_times.cpu().numpy() if hasattr(results.unbinding_times, 'cpu') else results.unbinding_times
            
            # Plot 1: Binding time histogram
            if binding_times is not None:
                axes[0, 0].hist(binding_times, bins=20, alpha=0.7, color='blue', density=True)
                axes[0, 0].set_xlabel('Binding Time (s)')
                axes[0, 0].set_ylabel('Density')
                axes[0, 0].set_title('Binding Time Distribution')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Unbinding time histogram
            if unbinding_times is not None:
                axes[0, 1].hist(unbinding_times, bins=20, alpha=0.7, color='red', density=True)
                axes[0, 1].set_xlabel('Unbinding Time (s)')
                axes[0, 1].set_ylabel('Density')
                axes[0, 1].set_title('Unbinding Time Distribution')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Log-log survival curves
            if binding_times is not None and unbinding_times is not None:
                # Binding survival
                sorted_binding = np.sort(binding_times)
                survival_binding = 1 - np.arange(1, len(sorted_binding)+1) / len(sorted_binding)
                axes[0, 2].loglog(sorted_binding, survival_binding, 'b-', label='Binding')
                
                # Unbinding survival
                sorted_unbinding = np.sort(unbinding_times)
                survival_unbinding = 1 - np.arange(1, len(sorted_unbinding)+1) / len(sorted_unbinding)
                axes[0, 2].loglog(sorted_unbinding, survival_unbinding, 'r-', label='Unbinding')
                
                axes[0, 2].set_xlabel('Time (s)')
                axes[0, 2].set_ylabel('Survival Probability')
                axes[0, 2].set_title('Survival Curves')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
            
            # Plot 4: Q-Q plot for exponential distribution (binding)
            if binding_times is not None and len(binding_times) > 5:
                from scipy import stats
                sorted_binding = np.sort(binding_times)
                lambda_est = 1 / np.mean(binding_times)
                theoretical_quantiles = stats.expon.ppf(np.linspace(0.01, 0.99, len(sorted_binding)), scale=1/lambda_est)
                
                axes[1, 0].plot(theoretical_quantiles, sorted_binding, 'bo', alpha=0.6)
                axes[1, 0].plot([min(theoretical_quantiles), max(theoretical_quantiles)], 
                               [min(theoretical_quantiles), max(theoretical_quantiles)], 'r--')
                axes[1, 0].set_xlabel('Theoretical Quantiles')
                axes[1, 0].set_ylabel('Sample Quantiles')
                axes[1, 0].set_title('Q-Q Plot: Binding Times vs Exponential')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Q-Q plot for exponential distribution (unbinding)
            if unbinding_times is not None and len(unbinding_times) > 5:
                from scipy import stats
                sorted_unbinding = np.sort(unbinding_times)
                lambda_est = 1 / np.mean(unbinding_times)
                theoretical_quantiles = stats.expon.ppf(np.linspace(0.01, 0.99, len(sorted_unbinding)), scale=1/lambda_est)
                
                axes[1, 1].plot(theoretical_quantiles, sorted_unbinding, 'ro', alpha=0.6)
                axes[1, 1].plot([min(theoretical_quantiles), max(theoretical_quantiles)], 
                               [min(theoretical_quantiles), max(theoretical_quantiles)], 'r--')
                axes[1, 1].set_xlabel('Theoretical Quantiles')
                axes[1, 1].set_ylabel('Sample Quantiles')
                axes[1, 1].set_title('Q-Q Plot: Unbinding Times vs Exponential')
                axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Rate estimation convergence
            if binding_times is not None and unbinding_times is not None and len(binding_times) > 10:
                n_points = min(50, len(binding_times))
                sample_sizes = np.linspace(5, len(binding_times), n_points).astype(int)
                
                kon_estimates = []
                koff_estimates = []
                
                for n in sample_sizes:
                    # Estimate kon
                    mean_binding = np.mean(binding_times[:n])
                    kon_est = 1 / (mean_binding * 1e-6)  # Assume 1 μM concentration
                    kon_estimates.append(kon_est)
                    
                    # Estimate koff
                    if n <= len(unbinding_times):
                        mean_unbinding = np.mean(unbinding_times[:n])
                        koff_est = 1 / mean_unbinding
                        koff_estimates.append(koff_est)
                    else:
                        koff_estimates.append(koff_estimates[-1] if koff_estimates else 0)
                
                axes[1, 2].semilogx(sample_sizes, kon_estimates, 'b-', label='kon')
                axes[1, 2].semilogx(sample_sizes[:len(koff_estimates)], koff_estimates, 'r-', label='koff')
                axes[1, 2].set_xlabel('Sample Size')
                axes[1, 2].set_ylabel('Rate Estimate')
                axes[1, 2].set_title('Rate Estimation Convergence')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Distribution analysis plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create distribution analysis plot: {e}")
            return fig
    
    def plot_selectivity_analysis(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot kinetic selectivity analysis
        
        Args:
            results: Dictionary with selectivity data
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating selectivity analysis plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Kinetic Selectivity Analysis', fontsize=16)
        
        try:
            # Extract selectivity data
            kinetic_selectivity = results.get('kinetic_selectivity', {})
            
            if not kinetic_selectivity:
                # Create dummy data for demonstration
                kinetic_selectivity = {
                    'kon_selectivity_ref1': 2.5,
                    'koff_selectivity_ref1': 0.8,
                    'residence_selectivity_ref1': 3.1,
                    'kon_selectivity_ref2': 1.2,
                    'koff_selectivity_ref2': 1.8,
                    'residence_selectivity_ref2': 0.7
                }
            
            # Parse selectivity data
            kon_sel = {k.replace('kon_selectivity_', ''): v for k, v in kinetic_selectivity.items() if 'kon_selectivity' in k}
            koff_sel = {k.replace('koff_selectivity_', ''): v for k, v in kinetic_selectivity.items() if 'koff_selectivity' in k}
            res_sel = {k.replace('residence_selectivity_', ''): v for k, v in kinetic_selectivity.items() if 'residence_selectivity' in k}
            
            # Plot 1: kon selectivity
            if kon_sel:
                refs = list(kon_sel.keys())
                values = list(kon_sel.values())
                axes[0, 0].bar(refs, values, alpha=0.7, color='blue')
                axes[0, 0].set_ylabel('kon Selectivity')
                axes[0, 0].set_title('Association Rate Selectivity')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No selectivity')
                axes[0, 0].legend()
            
            # Plot 2: koff selectivity
            if koff_sel:
                refs = list(koff_sel.keys())
                values = list(koff_sel.values())
                axes[0, 1].bar(refs, values, alpha=0.7, color='red')
                axes[0, 1].set_ylabel('koff Selectivity')
                axes[0, 1].set_title('Dissociation Rate Selectivity')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No selectivity')
                axes[0, 1].legend()
            
            # Plot 3: Residence time selectivity
            if res_sel:
                refs = list(res_sel.keys())
                values = list(res_sel.values())
                axes[1, 0].bar(refs, values, alpha=0.7, color='green')
                axes[1, 0].set_ylabel('Residence Time Selectivity')
                axes[1, 0].set_title('Residence Time Selectivity')
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No selectivity')
                axes[1, 0].legend()
            
            # Plot 4: Overall selectivity radar chart
            if kon_sel and koff_sel and res_sel:
                # Prepare data for radar chart
                refs = list(set(kon_sel.keys()) & set(koff_sel.keys()) & set(res_sel.keys()))
                if refs:
                    categories = ['kon', 'koff', 'residence_time']
                    
                    # Calculate angles for radar chart
                    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                    angles += angles[:1]  # Complete the circle
                    
                    # Plot for each reference
                    colors = ['blue', 'red', 'green', 'orange', 'purple']
                    for i, ref in enumerate(refs[:5]):  # Limit to 5 references
                        values = [
                            kon_sel.get(ref, 1),
                            koff_sel.get(ref, 1),
                            res_sel.get(ref, 1)
                        ]
                        values += values[:1]  # Complete the circle
                        
                        axes[1, 1].plot(angles, values, 'o-', linewidth=2, 
                                       label=ref, color=colors[i % len(colors)])
                        axes[1, 1].fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
                    
                    axes[1, 1].set_xticks(angles[:-1])
                    axes[1, 1].set_xticklabels(categories)
                    axes[1, 1].set_ylim(0, max(3, max([max(kon_sel.values()), max(koff_sel.values()), max(res_sel.values())])))
                    axes[1, 1].set_title('Selectivity Radar Chart')
                    axes[1, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                    axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Selectivity analysis plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create selectivity analysis plot: {e}")
            return fig
    
    def plot_convergence_analysis(
        self,
        results: Any,  # SimulationResults
        analysis: Dict[str, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot convergence analysis of simulation
        
        Args:
            results: SimulationResults object
            analysis: Analysis results dictionary
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating convergence analysis plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Simulation Convergence Analysis', fontsize=16)
        
        try:
            # Plot 1: Cumulative average of kinetic parameters
            if hasattr(results, 'binding_times') and len(results.binding_times) > 5:
                binding_times = results.binding_times.cpu().numpy() if hasattr(results.binding_times, 'cpu') else results.binding_times
                
                # Calculate cumulative averages
                cumulative_avg = np.cumsum(binding_times) / np.arange(1, len(binding_times)+1)
                
                axes[0, 0].plot(range(1, len(cumulative_avg)+1), cumulative_avg, 'b-')
                axes[0, 0].axhline(y=np.mean(binding_times), color='red', linestyle='--', 
                                  label=f'Final avg: {np.mean(binding_times):.2e}')
                axes[0, 0].set_xlabel('Sample Number')
                axes[0, 0].set_ylabel('Cumulative Average Binding Time (s)')
                axes[0, 0].set_title('Binding Time Convergence')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Standard error convergence
            if hasattr(results, 'binding_times') and len(results.binding_times) > 10:
                binding_times = results.binding_times.cpu().numpy() if hasattr(results.binding_times, 'cpu') else results.binding_times
                
                n_points = min(50, len(binding_times))
                sample_sizes = np.linspace(10, len(binding_times), n_points).astype(int)
                
                std_errors = []
                for n in sample_sizes:
                    subset = binding_times[:n]
                    std_error = np.std(subset) / np.sqrt(n)
                    std_errors.append(std_error)
                
                axes[0, 1].loglog(sample_sizes, std_errors, 'g-', marker='o')
                axes[0, 1].loglog(sample_sizes, 1/np.sqrt(sample_sizes), 'r--', 
                                 label='1/√n theoretical')
                axes[0, 1].set_xlabel('Sample Size')
                axes[0, 1].set_ylabel('Standard Error')
                axes[0, 1].set_title('Standard Error Convergence')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Event rate stability
            if (hasattr(results, 'binding_times') and len(results.binding_times) > 20 and
                hasattr(results, 'unbinding_times') and len(results.unbinding_times) > 20):
                
                binding_times = results.binding_times.cpu().numpy() if hasattr(results.binding_times, 'cpu') else results.binding_times
                unbinding_times = results.unbinding_times.cpu().numpy() if hasattr(results.unbinding_times, 'cpu') else results.unbinding_times
                
                # Calculate rates in sliding windows
                window_size = max(10, len(binding_times) // 10)
                binding_rates = []
                unbinding_rates = []
                window_centers = []
                
                for i in range(window_size, len(binding_times)):
                    window_binding = binding_times[i-window_size:i]
                    window_time = window_binding[-1] - window_binding[0]
                    if window_time > 0:
                        rate = len(window_binding) / window_time
                        binding_rates.append(rate)
                        window_centers.append(i)
                
                for i in range(window_size, len(unbinding_times)):
                    window_unbinding = unbinding_times[i-window_size:i]
                    window_time = window_unbinding[-1] - window_unbinding[0]
                    if window_time > 0:
                        rate = len(window_unbinding) / window_time
                        unbinding_rates.append(rate)
                
                if binding_rates:
                    axes[1, 0].plot(window_centers, binding_rates, 'b-', alpha=0.7, label='Binding')
                if unbinding_rates:
                    axes[1, 0].plot(range(window_size, window_size + len(unbinding_rates)), 
                                   unbinding_rates, 'r-', alpha=0.7, label='Unbinding')
                
                axes[1, 0].set_xlabel('Event Number')
                axes[1, 0].set_ylabel('Event Rate (events/s)')
                axes[1, 0].set_title('Event Rate Stability')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Convergence metrics summary
            convergence_metrics = {
                'Binding Events': analysis.get('n_binding_events', 0),
                'Unbinding Events': analysis.get('n_unbinding_events', 0),
                'Active Transitions': analysis.get('n_active_transitions', 0),
                'Trajectory Length': analysis.get('mean_trajectory_length', 0)
            }
            
            # Normalize metrics for comparison
            metrics = list(convergence_metrics.keys())
            values = list(convergence_metrics.values())
            
            if values and max(values) > 0:
                normalized_values = [v / max(values) for v in values]
                
                bars = axes[1, 1].bar(metrics, normalized_values, alpha=0.7, 
                                     color=['blue', 'red', 'green', 'orange'])
                axes[1, 1].set_ylabel('Normalized Value')
                axes[1, 1].set_title('Convergence Metrics')
                axes[1, 1].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                    f'{value:.0f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Convergence analysis plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create convergence analysis plot: {e}")
            return fig

