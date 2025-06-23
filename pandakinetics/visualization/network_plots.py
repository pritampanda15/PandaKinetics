# =============================================================================
# pandakinetics/visualization/network_plots.py - Network Visualization
# =============================================================================

"""
Network visualization tools for PandaKinetics

Provides plotting functions for transition networks, energy landscapes,
and molecular connectivity graphs.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

# Import with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available - network plotting limited")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.info("Plotly not available - interactive plots disabled")

# Import shared types with fallback
try:
    from ..types import SimulationResults
    from ..core.networks import TransitionNetwork
except ImportError:
    # Create dummy classes
    class SimulationResults:
        def __init__(self):
            pass
    
    class TransitionNetwork:
        def __init__(self):
            pass


class NetworkPlotter:
    """
    Network visualization and plotting tools
    
    Provides methods for visualizing transition networks, energy landscapes,
    and molecular connectivity patterns.
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize network plotter
        
        Args:
            style: Matplotlib style ('seaborn', 'ggplot', etc.)
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        
        # Set plotting style
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default")
        
        # Set color palette
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
        logger.info(f"NetworkPlotter initialized with style '{style}'")
    
    def plot_transition_network(
        self,
        network: Any,  # TransitionNetwork
        results: Optional[Any] = None,  # SimulationResults
        layout: str = 'spring',
        node_size_by: str = 'population',
        edge_width_by: str = 'transitions',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot transition network graph
        
        Args:
            network: TransitionNetwork object
            results: SimulationResults for population/transition data
            layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai')
            node_size_by: What to scale node size by ('population', 'energy', 'uniform')
            edge_width_by: What to scale edge width by ('transitions', 'rates', 'uniform')
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        if not NETWORKX_AVAILABLE:
            logger.error("NetworkX required for network plotting")
            return None
        
        logger.info("Creating transition network plot...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        try:
            # Create NetworkX graph
            G = self._build_networkx_graph(network, results)
            
            # Choose layout
            if layout == 'spring':
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # Calculate node sizes
            node_sizes = self._calculate_node_sizes(G, node_size_by, results)
            
            # Calculate edge widths
            edge_widths = self._calculate_edge_widths(G, edge_width_by, results)
            
            # Calculate node colors (by energy if available)
            node_colors = self._calculate_node_colors(G, network)
            
            # Draw network
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_size=node_sizes,
                node_color=node_colors,
                cmap='viridis',
                alpha=0.8
            )
            
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                width=edge_widths,
                alpha=0.6,
                edge_color='gray'
            )
            
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            
            # Add colorbar for node colors
            if hasattr(network, 'energies'):
                sm = plt.cm.ScalarMappable(cmap='viridis')
                sm.set_array(node_colors)
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('Energy (kcal/mol)')
            
            ax.set_title('Transition Network')
            ax.axis('off')
            
            # Add legend
            self._add_network_legend(ax, node_size_by, edge_width_by)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Network plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create network plot: {e}")
            return fig
    
    def plot_energy_landscape(
        self,
        network: Any,  # TransitionNetwork
        results: Optional[Any] = None,  # SimulationResults
        projection: str = '2d',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot energy landscape
        
        Args:
            network: TransitionNetwork object
            results: SimulationResults for population data
            projection: '2d' or '3d' projection
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating energy landscape plot...")
        
        if projection == '3d':
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        try:
            # Get energy data
            if hasattr(network, 'energies'):
                energies = network.energies.cpu().numpy() if hasattr(network.energies, 'cpu') else network.energies
            else:
                # Create dummy energies
                n_states = 10
                energies = np.random.randn(n_states) * 5
            
            # Get population data
            if results and hasattr(results, 'state_populations'):
                populations = results.state_populations.cpu().numpy() if hasattr(results.state_populations, 'cpu') else results.state_populations
            else:
                populations = np.ones(len(energies))
            
            # Get positions for visualization
            positions = self._get_state_positions(network, len(energies))
            
            if projection == '3d':
                # 3D scatter plot
                scatter = ax.scatter(
                    positions[:, 0], positions[:, 1], energies,
                    s=populations * 100,
                    c=energies,
                    cmap='viridis',
                    alpha=0.8
                )
                ax.set_zlabel('Energy (kcal/mol)')
            else:
                # 2D scatter plot
                scatter = ax.scatter(
                    positions[:, 0], positions[:, 1],
                    s=populations * 100,
                    c=energies,
                    cmap='viridis',
                    alpha=0.8
                )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Energy (kcal/mol)')
            
            ax.set_xlabel('Coordinate 1')
            ax.set_ylabel('Coordinate 2')
            ax.set_title('Energy Landscape')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Energy landscape plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create energy landscape plot: {e}")
            return fig
    
    def plot_pathway_analysis(
        self,
        network: Any,  # TransitionNetwork
        results: Any,  # SimulationResults
        source_states: List[int],
        target_states: List[int],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot pathway analysis between source and target states
        
        Args:
            network: TransitionNetwork object
            results: SimulationResults object
            source_states: List of source state indices
            target_states: List of target state indices
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        if not NETWORKX_AVAILABLE:
            logger.error("NetworkX required for pathway analysis")
            return None
        
        logger.info("Creating pathway analysis plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        try:
            # Create NetworkX graph
            G = self._build_networkx_graph(network, results)
            
            # Find shortest paths
            paths = []
            for source in source_states:
                for target in target_states:
                    try:
                        path = nx.shortest_path(G, source, target)
                        paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
            
            # Plot 1: Network with highlighted paths
            pos = nx.spring_layout(G)
            
            # Draw base network
            nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightblue', alpha=0.5)
            nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3)
            
            # Highlight source and target states
            nx.draw_networkx_nodes(G, pos, nodelist=source_states, ax=ax1, 
                                 node_color='green', node_size=500, label='Source')
            nx.draw_networkx_nodes(G, pos, nodelist=target_states, ax=ax1,
                                 node_color='red', node_size=500, label='Target')
            
            # Highlight paths
            colors = ['orange', 'purple', 'brown', 'pink']
            for i, path in enumerate(paths[:4]):  # Show up to 4 paths
                path_edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax1,
                                     edge_color=colors[i % len(colors)], width=3)
            
            nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8)
            ax1.set_title('Pathway Analysis')
            ax1.axis('off')
            ax1.legend()
            
            # Plot 2: Path length distribution
            if paths:
                path_lengths = [len(path) - 1 for path in paths]
                ax2.hist(path_lengths, bins=max(1, len(set(path_lengths))), alpha=0.7)
                ax2.set_xlabel('Path Length (steps)')
                ax2.set_ylabel('Count')
                ax2.set_title('Path Length Distribution')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Pathway analysis plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create pathway analysis plot: {e}")
            return fig
    
    def plot_interactive_network(
        self,
        network: Any,  # TransitionNetwork
        results: Optional[Any] = None,  # SimulationResults
        save_path: Optional[str] = None
    ):
        """
        Create interactive network plot using Plotly
        
        Args:
            network: TransitionNetwork object
            results: SimulationResults object
            save_path: Path to save HTML file
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly required for interactive plots")
            return
        
        logger.info("Creating interactive network plot...")
        
        try:
            # Create NetworkX graph
            G = self._build_networkx_graph(network, results)
            
            # Get layout
            pos = nx.spring_layout(G)
            
            # Prepare node data
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_text = [f"State {node}" for node in G.nodes()]
            
            # Prepare edge data
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create traces
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='gray'),
                hoverinfo='none',
                mode='lines'
            )
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    colorscale='viridis',
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        xanchor="left",
                        titleside="right"
                    )
                )
            )
            
            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title='Interactive Transition Network',
                              titlefont_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="Interactive network visualization",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002 ) ],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Interactive network plot saved to {save_path}")
            else:
                fig.show()
            
        except Exception as e:
            logger.error(f"Failed to create interactive network plot: {e}")
    
    def _build_networkx_graph(self, network: Any, results: Optional[Any] = None):
        """Build NetworkX graph from transition network"""
        
        G = nx.Graph()
        
        # Add nodes
        if hasattr(network, 'n_states'):
            n_states = network.n_states
        elif hasattr(network, 'positions'):
            n_states = len(network.positions)
        else:
            n_states = 10  # Default fallback
        
        for i in range(n_states):
            G.add_node(i)
        
        # Add edges
        if hasattr(network, 'adjacency_matrix'):
            adj_matrix = network.adjacency_matrix.cpu().numpy() if hasattr(network.adjacency_matrix, 'cpu') else network.adjacency_matrix
            for i in range(n_states):
                for j in range(i+1, n_states):
                    if adj_matrix[i, j] > 0:
                        G.add_edge(i, j, weight=adj_matrix[i, j])
        elif results and hasattr(results, 'transition_counts'):
            trans_counts = results.transition_counts.cpu().numpy() if hasattr(results.transition_counts, 'cpu') else results.transition_counts
            for i in range(n_states):
                for j in range(i+1, n_states):
                    if trans_counts[i, j] > 0:
                        G.add_edge(i, j, weight=trans_counts[i, j])
        else:
            # Create random connectivity for demo
            for i in range(n_states):
                for j in range(i+1, min(i+3, n_states)):
                    G.add_edge(i, j, weight=1.0)
        
        return G
    
    def _calculate_node_sizes(self, G, size_by: str, results: Optional[Any] = None):
        """Calculate node sizes for visualization"""
        
        if size_by == 'population' and results and hasattr(results, 'state_populations'):
            populations = results.state_populations.cpu().numpy() if hasattr(results.state_populations, 'cpu') else results.state_populations
            # Normalize and scale
            sizes = (populations / np.max(populations)) * 500 + 100
            return sizes[:len(G.nodes())]
        elif size_by == 'degree':
            degrees = dict(G.degree())
            max_degree = max(degrees.values()) if degrees else 1
            return [(degrees.get(node, 1) / max_degree) * 500 + 100 for node in G.nodes()]
        else:
            return [200] * len(G.nodes())  # Uniform size
    
    def _calculate_edge_widths(self, G, width_by: str, results: Optional[Any] = None):
        """Calculate edge widths for visualization"""
        
        if width_by == 'transitions' and results and hasattr(results, 'transition_counts'):
            trans_counts = results.transition_counts.cpu().numpy() if hasattr(results.transition_counts, 'cpu') else results.transition_counts
            widths = []
            for edge in G.edges():
                i, j = edge
                count = trans_counts[i, j] if i < len(trans_counts) and j < len(trans_counts[0]) else 1
                widths.append(max(count / np.max(trans_counts) * 5, 0.5))
            return widths
        elif width_by == 'weight':
            weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
            max_weight = max(weights) if weights else 1
            return [(w / max_weight) * 5 + 0.5 for w in weights]
        else:
            return [1.0] * len(G.edges())  # Uniform width
    
    def _calculate_node_colors(self, G, network: Any):
        """Calculate node colors based on energy"""
        
        if hasattr(network, 'energies'):
            energies = network.energies.cpu().numpy() if hasattr(network.energies, 'cpu') else network.energies
            return energies[:len(G.nodes())]
        else:
            return list(range(len(G.nodes())))  # Default coloring
    
    def _get_state_positions(self, network: Any, n_states: int):
        """Get 2D positions for states"""
        
        if hasattr(network, 'positions'):
            positions = network.positions.cpu().numpy() if hasattr(network.positions, 'cpu') else network.positions
            # Use first two dimensions for 2D projection
            if positions.ndim == 3:  # (n_states, n_atoms, 3)
                return positions[:, 0, :2]  # Use first atom coordinates
            elif positions.ndim == 2:  # (n_states, n_dims)
                return positions[:, :2]
        
        # Generate random positions for visualization
        return np.random.randn(n_states, 2) * 5
    
    def _add_network_legend(self, ax, node_size_by: str, edge_width_by: str):
        """Add legend to network plot"""
        
        legend_text = f"Node size: {node_size_by}\nEdge width: {edge_width_by}"
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
