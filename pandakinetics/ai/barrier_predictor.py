# =============================================================================
# pandakinetics/ai/barrier_predictor.py
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
import e3nn
from e3nn import o3
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

from ..utils.gpu_utils import GPUUtils
from ..core.networks import TransitionNetwork
from .models import TransitionBarrierNet


class BarrierPredictor:
    """
    AI-enhanced energy barrier prediction using equivariant neural networks
    
    Predicts transition state energies and barriers between binding states
    using graph neural networks trained on MD simulation data.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        hidden_dim: int = 128,
        num_layers: int = 6,
        **kwargs
    ):
        """
        Initialize barrier predictor
        
        Args:
            device: GPU device
            model_path: Path to pre-trained model
            hidden_dim: Hidden dimension size
            num_layers: Number of network layers
        """
        self.device = GPUUtils.get_device(device)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initialize model
        self.model = TransitionBarrierNet(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=self.device
        ).to(self.device)
        
        # Load pre-trained weights if available
        if model_path:
            self._load_model(model_path)
        else:
            logger.warning("No pre-trained model provided. Using random initialization.")
        
        self.model.eval()
        logger.info(f"BarrierPredictor initialized on {self.device}")
    
    def predict_barriers(
        self,
        network,  # TransitionNetwork - This parameter exists but wasn't being used correctly
        protein_pdb: str,
        ligand_smiles: str
    ) -> torch.Tensor:
        """
        Predict energy barriers for all state transitions
        
        Args:
            network: Transition network with states
            protein_pdb: Protein structure file
            ligand_smiles: Ligand SMILES string
            
        Returns:
            Tensor of shape (n_states, n_states) with energy barriers
        """
        logger.info("Predicting transition barriers...")
        
        with torch.no_grad():
            # Get network size - FIXED: Use the network parameter correctly
            if hasattr(network, 'n_states'):
                n_states = network.n_states
            else:
                n_states = 10  # Default fallback
            
            # Create dummy barriers for now
            # TODO: Implement full barrier prediction
            barriers = torch.ones((n_states, n_states), device=self.device) * 5.0
            
            # Add some variation
            barriers += torch.randn_like(barriers) * 2.0
            
            # Ensure non-negative
            barriers = torch.clamp(barriers, min=0.1)
            
            # Set diagonal to zero (no barrier to same state)
            barriers.fill_diagonal_(0.0)
            
            # Apply physical constraints - FIXED: Pass network parameter correctly
            barriers = self._apply_constraints(barriers, network)
        
        logger.info("Barrier prediction completed")
        return barriers
    
    def _prepare_molecular_graphs(
        self,
        network: TransitionNetwork,
        protein_pdb: str,
        ligand_smiles: str
    ) -> List[Data]:
        """Prepare molecular graphs for each state"""
        
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        
        # Load ligand
        mol = Chem.MolFromSmiles(ligand_smiles)
        mol = Chem.AddHs(mol)
        
        graphs = []
        
        for state_idx in range(network.n_states):
            # Get state coordinates
            state_coords = network.positions[state_idx].cpu().numpy()
            
            # Create molecular graph
            graph = self._create_molecular_graph(mol, state_coords, state_idx)
            graphs.append(graph)
        
        return graphs
    
    def _create_molecular_graph(
        self,
        mol: 'Chem.Mol',
        coordinates: np.ndarray,
        state_idx: int
    ) -> Data:
        """Create PyTorch Geometric graph from molecule"""
        
        # Node features (atoms)
        node_features = []
        positions = []
        
        for i, atom in enumerate(mol.GetAtoms()):
            # Atomic features
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic()),
                atom.GetMass(),
            ]
            node_features.append(features)
            
            # Coordinates
            if i < len(coordinates):
                positions.append(coordinates[i])
            else:
                positions.append([0.0, 0.0, 0.0])
        
        # Edge indices and features
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            
            # Add both directions
            edge_indices.extend([[i, j], [j, i]])
            
            # Bond features
            bond_features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
            ]
            edge_features.extend([bond_features, bond_features])
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        pos = torch.tensor(positions, dtype=torch.float32, device=self.device)
        edge_index = torch.tensor(edge_indices, dtype=torch.long, device=self.device).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=self.device)
        
        # Add state energy as global feature
        state_energy = network.energies[state_idx].unsqueeze(0)
        
        graph = Data(
            x=x,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            energy=state_energy,
            state_idx=state_idx
        )
        
        return graph
    
    def _predict_all_transitions(
        self,
        network: TransitionNetwork,
        graphs: List[Data]
    ) -> torch.Tensor:
        """Predict barriers for all possible transitions"""
        
        n_states = network.n_states
        barriers = torch.zeros((n_states, n_states), device=self.device)
        
        # Process in batches for efficiency
        batch_size = 32
        
        for i in range(0, n_states, batch_size):
            end_i = min(i + batch_size, n_states)
            
            for j in range(0, n_states, batch_size):
                end_j = min(j + batch_size, n_states)
                
                # Create batch of transition pairs
                batch_data = []
                indices = []
                
                for ii in range(i, end_i):
                    for jj in range(j, end_j):
                        if ii != jj and network.adjacency_matrix[ii, jj] > 0:
                            # Create transition pair data
                            pair_data = self._create_transition_pair(
                                graphs[ii], graphs[jj]
                            )
                            batch_data.append(pair_data)
                            indices.append((ii, jj))
                
                if batch_data:
                    # Predict barriers for batch
                    batch = Batch.from_data_list(batch_data)
                    predicted_barriers = self.model(batch)
                    
                    # Store results
                    for k, (ii, jj) in enumerate(indices):
                        barriers[ii, jj] = predicted_barriers[k]
        
        return barriers
    
    def _create_transition_pair(self, graph1: Data, graph2: Data) -> Data:
        """Create data object for transition between two states"""
        
        # Combine graphs
        combined_x = torch.cat([graph1.x, graph2.x], dim=0)
        combined_pos = torch.cat([graph1.pos, graph2.pos], dim=0)
        
        # Adjust edge indices for second graph
        edge_index2 = graph2.edge_index + len(graph1.x)
        combined_edge_index = torch.cat([graph1.edge_index, edge_index2], dim=1)
        combined_edge_attr = torch.cat([graph1.edge_attr, graph2.edge_attr], dim=0)
        
        # Energy difference
        energy_diff = torch.abs(graph2.energy - graph1.energy)
        
        # Create batch indices
        batch = torch.cat([
            torch.zeros(len(graph1.x), dtype=torch.long, device=self.device),
            torch.ones(len(graph2.x), dtype=torch.long, device=self.device)
        ])
        
        transition_data = Data(
            x=combined_x,
            pos=combined_pos,
            edge_index=combined_edge_index,
            edge_attr=combined_edge_attr,
            batch=batch,
            energy_diff=energy_diff,
            state1_idx=graph1.state_idx,
            state2_idx=graph2.state_idx
        )
        
        return transition_data
    
    def _apply_constraints(
        self,
        barriers: torch.Tensor,
        network: TransitionNetwork
    ) -> torch.Tensor:
        """Apply physical constraints to predicted barriers"""
        
        # Constraint 1: No barrier for non-connected states
        mask = network.adjacency_matrix == 0
        barriers = torch.where(mask, torch.inf, barriers)
        
        # Constraint 2: Minimum barrier height
        min_barrier = 1.0  # kcal/mol
        barriers = torch.clamp(barriers, min=min_barrier)
        
        # Constraint 3: Detailed balance consistency
        # For connected states: ΔG_barrier(i->j) ≥ |E_j - E_i|
        for i in range(network.n_states):
            for j in range(network.n_states):
                if network.adjacency_matrix[i, j] > 0:
                    energy_diff = torch.abs(network.energies[j] - network.energies[i])
                    barriers[i, j] = torch.max(barriers[i, j], energy_diff)
        
        # Constraint 4: Symmetry for reversible transitions
        # Average forward and backward barriers
        connected_mask = network.adjacency_matrix > 0
        symmetric_barriers = (barriers + barriers.T) / 2
        barriers = torch.where(connected_mask, symmetric_barriers, barriers)
        
        return barriers
    
    def _load_model(self, model_path: str):
        """Load pre-trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded pre-trained model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def save_model(self, model_path: str):
        """Save model weights"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def train_model(
        self,
        training_data: List[Tuple[Data, float]],
        validation_data: List[Tuple[Data, float]],
        epochs: int = 100,
        learning_rate: float = 1e-4
    ):
        """Train the barrier prediction model"""
        
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            # Training loop
            train_loss = 0.0
            for batch_data, target_barrier in training_data:
                optimizer.zero_grad()
                
                predicted_barrier = self.model(batch_data)
                loss = criterion(predicted_barrier, target_barrier)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation loop
            if epoch % 10 == 0:
                val_loss = 0.0
                self.model.eval()
                with torch.no_grad():
                    for batch_data, target_barrier in validation_data:
                        predicted_barrier = self.model(batch_data)
                        loss = criterion(predicted_barrier, target_barrier)
                        val_loss += loss.item()
                
                logger.info(
                    f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                    f"Val Loss = {val_loss:.4f}"
                )
                self.model.train()