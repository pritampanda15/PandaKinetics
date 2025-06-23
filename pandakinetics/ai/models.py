# =============================================================================
# pandakinetics/ai/models.py - Neural Network Models
# =============================================================================

"""
Neural network models for PandaKinetics AI components

This module contains the neural network architectures used for:
- Transition barrier prediction
- Enhanced sampling
- Molecular property prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import math

# Try to import advanced libraries, with fallbacks
try:
    from torch_geometric.nn import MessagePassing, global_mean_pool
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    # Create dummy classes for compatibility
    class MessagePassing:
        def __init__(self, *args, **kwargs):
            pass
    
    class Data:
        def __init__(self, *args, **kwargs):
            pass
    
    def global_mean_pool(x, batch):
        return torch.mean(x, dim=0, keepdim=True)

try:
    import e3nn
    from e3nn import o3
    from e3nn.nn import Gate
    E3NN_AVAILABLE = True
except ImportError:
    E3NN_AVAILABLE = False
    # Create dummy classes
    class o3:
        class Irreps:
            def __init__(self, *args, **kwargs):
                pass
        
        class Linear:
            def __init__(self, *args, **kwargs):
                pass
        
        class TensorProduct:
            def __init__(self, *args, **kwargs):
                pass

from loguru import logger


class TransitionBarrierNet(nn.Module):
    """
    Neural network for predicting transition barriers between molecular states
    
    This model can work with or without advanced geometric libraries.
    Falls back to standard PyTorch implementations when needed.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
        max_radius: float = 5.0,
        num_basis: int = 8,
        device: Optional[str] = None,
        use_geometric: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_radius = max_radius
        self.num_basis = num_basis
        self.device = device
        self.use_geometric = use_geometric and TORCH_GEOMETRIC_AVAILABLE
        
        if self.use_geometric:
            self._build_geometric_model()
        else:
            self._build_standard_model()
            logger.warning("Using standard model - install torch-geometric for better performance")
    
    def _build_geometric_model(self):
        """Build model using torch-geometric (preferred)"""
        
        # Node feature dimensions
        self.node_dim = 6  # atomic features
        self.edge_dim = 3  # edge features
        
        # Input embeddings
        self.node_embedding = nn.Linear(self.node_dim, self.hidden_dim)
        self.edge_embedding = nn.Linear(self.edge_dim + self.num_basis, self.hidden_dim)
        
        # Message passing layers
        self.message_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = GraphMessagePassing(
                hidden_dim=self.hidden_dim,
                edge_dim=self.hidden_dim
            )
            self.message_layers.append(layer)
        
        # Output layers for barrier prediction
        self.barrier_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # *2 for two states
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive barriers
        )
        
        # Radial basis functions
        self.distance_expansion = GaussianBasis(
            start=0.0, stop=self.max_radius, num_basis=self.num_basis
        )
    
    def _build_standard_model(self):
        """Build standard model without geometric dependencies"""
        
        # Simple feedforward network for molecular descriptors
        self.input_dim = 50  # Molecular descriptor dimension
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Barrier prediction head
        self.barrier_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # *2 for two states
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive barriers
        )
    
    def forward(self, data) -> torch.Tensor:
        """
        Forward pass for barrier prediction
        
        Args:
            data: Input data (format depends on model type)
            
        Returns:
            Predicted energy barriers
        """
        if self.use_geometric:
            return self._forward_geometric(data)
        else:
            return self._forward_standard(data)
    
    def _forward_geometric(self, data) -> torch.Tensor:
        """Forward pass for geometric model"""
        
        if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
            # Fallback to standard model
            return self._forward_standard(data)
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Embed node features
        x = self.node_embedding(x)
        
        # Calculate edge features with distance expansion
        if hasattr(data, 'pos'):
            edge_vec = data.pos[edge_index[1]] - data.pos[edge_index[0]]
            edge_length = torch.norm(edge_vec, dim=1)
            edge_length_embedded = self.distance_expansion(edge_length)
            
            if edge_attr is not None:
                edge_features = torch.cat([edge_attr, edge_length_embedded], dim=1)
            else:
                edge_features = edge_length_embedded
        else:
            edge_features = edge_attr if edge_attr is not None else torch.zeros(edge_index.size(1), self.edge_dim, device=x.device)
        
        edge_features = self.edge_embedding(edge_features)
        
        # Message passing
        for layer in self.message_layers:
            x = layer(x, edge_index, edge_features)
        
        # Global pooling for each state in the transition pair
        if batch is not None:
            # Separate states in batch
            unique_batches = torch.unique(batch)
            if len(unique_batches) >= 2:
                state1_mask = batch == unique_batches[0]
                state2_mask = batch == unique_batches[1]
                
                state1_repr = global_mean_pool(x[state1_mask], torch.zeros(torch.sum(state1_mask), dtype=torch.long, device=x.device))
                state2_repr = global_mean_pool(x[state2_mask], torch.zeros(torch.sum(state2_mask), dtype=torch.long, device=x.device))
                
                combined_repr = torch.cat([state1_repr, state2_repr], dim=1)
            else:
                # Single state case
                pooled = global_mean_pool(x, batch)
                combined_repr = torch.cat([pooled, pooled], dim=1)
        else:
            # No batch information
            pooled = torch.mean(x, dim=0, keepdim=True)
            combined_repr = torch.cat([pooled, pooled], dim=1)
        
        # Predict barrier
        barrier = self.barrier_head(combined_repr)
        return barrier.squeeze(-1)
    
    def _forward_standard(self, data) -> torch.Tensor:
        """Forward pass for standard model"""
        
        # Extract molecular descriptors
        if isinstance(data, dict):
            # Handle dictionary input
            if 'descriptors' in data:
                features = data['descriptors']
            elif 'features' in data:
                features = data['features']
            else:
                # Create dummy features
                features = torch.randn(1, self.input_dim, device=next(self.parameters()).device)
        elif isinstance(data, torch.Tensor):
            features = data
        else:
            # Fallback to dummy features
            features = torch.randn(1, self.input_dim, device=next(self.parameters()).device)
        
        # Ensure correct shape
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # Extract features for each state
        batch_size = features.size(0)
        if batch_size >= 2:
            state1_features = self.feature_extractor(features[:batch_size//2])
            state2_features = self.feature_extractor(features[batch_size//2:])
        else:
            # Single state case - duplicate
            state_features = self.feature_extractor(features)
            state1_features = state_features
            state2_features = state_features
        
        # Pool features
        state1_repr = torch.mean(state1_features, dim=0, keepdim=True)
        state2_repr = torch.mean(state2_features, dim=0, keepdim=True)
        
        # Combine representations
        combined_repr = torch.cat([state1_repr, state2_repr], dim=1)
        
        # Predict barrier
        barrier = self.barrier_head(combined_repr)
        return barrier.squeeze(-1)


class GraphMessagePassing(MessagePassing):
    """Message passing layer for molecular graphs"""
    
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr='add', node_dim=0)
        
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        # Message function
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, edge_index, edge_attr):
        """Forward pass"""
        
        # Save input for residual connection
        x_in = x
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Update
        out = self.update_net(torch.cat([x, out], dim=1))
        
        # Residual connection and normalization
        out = self.norm(out + x_in)
        
        return out
    
    def message(self, x_i, x_j, edge_attr):
        """Compute messages"""
        
        # Concatenate node features and edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        
        # Compute message
        msg = self.message_net(msg_input)
        
        return msg


class GaussianBasis(nn.Module):
    """Gaussian radial basis functions for distance encoding"""
    
    def __init__(self, start: float, stop: float, num_basis: int):
        super().__init__()
        
        self.num_basis = num_basis
        self.start = start
        self.stop = stop
        
        # Gaussian centers and widths
        centers = torch.linspace(start, stop, num_basis)
        self.register_buffer('centers', centers)
        
        width = (stop - start) / (num_basis - 1)
        self.register_buffer('width', torch.tensor(width))
    
    def forward(self, distances):
        """Expand distances using Gaussian basis"""
        
        # distances: (num_edges,)
        # centers: (num_basis,)
        
        diff = distances.unsqueeze(-1) - self.centers.unsqueeze(0)
        basis = torch.exp(-0.5 * (diff / self.width) ** 2)
        
        return basis


class MolecularEncoder(nn.Module):
    """Encoder for molecular representations"""
    
    def __init__(
        self,
        input_dim: int = 50,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """Encode molecular features"""
        return self.encoder(x)


class EnergyPredictor(nn.Module):
    """Simple energy predictor for molecular states"""
    
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.predictor = nn.Sequential(*layers)
    
    def forward(self, x):
        """Predict energy"""
        return self.predictor(x)


class KineticPredictor(nn.Module):
    """Predictor for kinetic parameters (kon, koff)"""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()
        
        # Shared layers
        shared_layers = []
        shared_layers.append(nn.Linear(input_dim, hidden_dim))
        shared_layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(0.1))
        
        self.shared_net = nn.Sequential(*shared_layers)
        
        # Separate heads for kon and koff
        self.kon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive
        )
        
        self.koff_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive
        )
    
    def forward(self, x):
        """Predict kon and koff"""
        
        # Shared representation
        shared_repr = self.shared_net(x)
        
        # Separate predictions
        kon = self.kon_head(shared_repr)
        koff = self.koff_head(shared_repr)
        
        return {
            'kon': kon.squeeze(-1),
            'koff': koff.squeeze(-1)
        }


def create_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to create models"""
    
    models = {
        'transition_barrier': TransitionBarrierNet,
        'molecular_encoder': MolecularEncoder,
        'energy_predictor': EnergyPredictor,
        'kinetic_predictor': KineticPredictor
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](**kwargs)


def get_model_info():
    """Get information about available models and dependencies"""
    
    info = {
        'torch_geometric_available': TORCH_GEOMETRIC_AVAILABLE,
        'e3nn_available': E3NN_AVAILABLE,
        'available_models': [
            'transition_barrier',
            'molecular_encoder', 
            'energy_predictor',
            'kinetic_predictor'
        ]
    }
    
    return info


# Model registry for easy access
MODEL_REGISTRY = {
    'TransitionBarrierNet': TransitionBarrierNet,
    'MolecularEncoder': MolecularEncoder,
    'EnergyPredictor': EnergyPredictor,
    'KineticPredictor': KineticPredictor
}

# Export main classes
__all__ = [
    'TransitionBarrierNet',
    'MolecularEncoder',
    'EnergyPredictor', 
    'KineticPredictor',
    'GraphMessagePassing',
    'GaussianBasis',
    'create_model',
    'get_model_info',
    'MODEL_REGISTRY'
]
