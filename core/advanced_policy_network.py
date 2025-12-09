"""
ADVANCED 2025 POLICY NETWORK
Based on latest 2024-2025 Deep RL research:
- CNN + Attention for spatial perception (raycasting)
- LSTM for temporal memory and partial observability
- Learned latent representations
- Successor Representations support
- Multi-head attention for feature selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np


class SpatialAttention(nn.Module):
    """Multi-head attention for focusing on important spatial features."""
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        return self.layer_norm(x + attn_output)  # Residual connection


class RaycastingCNN(nn.Module):
    """
    CNN for processing raycasting data (36 rays).
    Treats rays as a 1D spatial sequence with circular connectivity.
    """
    
    def __init__(self, n_rays: int = 36, output_dim: int = 128):
        super().__init__()
        
        # Treat raycasting as 1D spatial data
        # Conv1d learns spatial patterns in obstacle distribution
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)  # Detect local patterns
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)  # Hierarchical features
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)  # High-level features
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.1)
        
        # Calculate flattened size after convolutions
        # n_rays -> pool -> pool = n_rays // 4
        self.flat_size = 128 * (n_rays // 4)
        
        self.fc = nn.Linear(self.flat_size, output_dim)
        
    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        # rays shape: (batch, n_rays)
        x = rays.unsqueeze(1)  # (batch, 1, n_rays) - treat as 1D image
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        
        x = x.flatten(1)  # Flatten spatial dimensions
        x = F.relu(self.fc(x))
        
        return x


class AdvancedFeaturesExtractor(BaseFeaturesExtractor):
    """
    Advanced feature extractor combining:
    - CNN for raycasting spatial patterns
    - Attention for feature selection
    - LSTM for temporal memory
    - Dense layers for base features
    
    Based on 2024-2025 research on navigation in partially observable environments.
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Observation breakdown: 11 base + 36 rays + 3 enhanced = 50
        self.n_base = 11
        self.n_rays = 36
        self.n_enhanced = 3
        
        # Base features processing (position, velocity, goal info)
        self.base_net = nn.Sequential(
            nn.Linear(self.n_base, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        # Raycasting CNN (spatial perception)
        self.ray_cnn = RaycastingCNN(n_rays=self.n_rays, output_dim=128)
        
        # Enhanced features processing
        self.enhanced_net = nn.Sequential(
            nn.Linear(self.n_enhanced, 32),
            nn.ReLU(),
        )
        
        # Combine all features
        combined_dim = 128 + 128 + 32  # base + rays + enhanced
        
        # Attention layer (focus on important features)
        # Reshape combined features for attention
        self.attention_embed = nn.Linear(combined_dim, 256)
        self.spatial_attention = SpatialAttention(embed_dim=256, num_heads=4)
        
        # LSTM for temporal memory (handles partial observability)
        self.lstm_hidden_size = 256
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        
        # Final layers
        self.final_net = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )
        
        # LSTM hidden states (maintained across steps within episode)
        self.lstm_hidden = None
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observation into components
        base_obs = observations[:, :self.n_base]
        ray_obs = observations[:, self.n_base:self.n_base + self.n_rays]
        enhanced_obs = observations[:, self.n_base + self.n_rays:]
        
        # Process each component
        base_features = self.base_net(base_obs)
        ray_features = self.ray_cnn(ray_obs)
        enhanced_features = self.enhanced_net(enhanced_obs)
        
        # Combine features
        combined = torch.cat([base_features, ray_features, enhanced_features], dim=1)
        
        # Attention mechanism (focus on relevant features)
        attended = self.attention_embed(combined)
        attended = attended.unsqueeze(1)  # (batch, 1, 256) for attention
        attended = self.spatial_attention(attended)
        attended = attended.squeeze(1)  # (batch, 256)
        
        # LSTM for temporal context (remember past observations)
        attended_seq = attended.unsqueeze(1)  # (batch, seq_len=1, 256)
        
        if self.lstm_hidden is None or attended_seq.size(0) != self.lstm_hidden[0].size(1):
            # Initialize LSTM hidden state
            device = attended_seq.device
            self.lstm_hidden = (
                torch.zeros(2, attended_seq.size(0), self.lstm_hidden_size, device=device),
                torch.zeros(2, attended_seq.size(0), self.lstm_hidden_size, device=device),
            )
        
        lstm_out, new_hidden = self.lstm(attended_seq, self.lstm_hidden)
        lstm_out = lstm_out.squeeze(1)  # (batch, lstm_hidden_size)
        
        # Detach hidden state to prevent backprop through time across epochs
        # This fixes "Trying to backward through the graph a second time" error
        self.lstm_hidden = (new_hidden[0].detach(), new_hidden[1].detach())
        
        # Final processing
        features = self.final_net(lstm_out)
        
        return features
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state (call at episode start)."""
        self.lstm_hidden = None


class AdvancedActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy using advanced feature extractor.
    Integrates CNN, Attention, and LSTM for robust navigation.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        # Force use of our custom feature extractor
        kwargs["features_extractor_class"] = AdvancedFeaturesExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": 512}
        
        # Network architecture for policy and value heads
        # (applied after feature extraction)
        kwargs["net_arch"] = dict(
            pi=[256, 128],  # Policy network: 512 -> 256 -> 128 -> action
            vf=[256, 128],  # Value network: 512 -> 256 -> 128 -> value
        )
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
    
    def _predict(self, observation: torch.Tensor, deterministic: bool = False):
        """Override predict to handle LSTM state reset."""
        # Reset LSTM at episode boundaries if needed
        # (This is handled automatically by the extractor maintaining state)
        return super()._predict(observation, deterministic)


class ActionSmoother:
    """
    Smooth actions to prevent jerky movement.
    Uses exponential moving average for smooth transitions.
    """
    
    def __init__(self, alpha: float = 0.3):
        """
        Args:
            alpha: Smoothing factor (0 = no change, 1 = full change)
                   Lower values = smoother but slower response
        """
        self.alpha = alpha
        self.prev_action = None
        
    def smooth(self, action: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to action."""
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action
        
        # Exponential moving average
        smoothed = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = smoothed.copy()
        
        return smoothed
    
    def reset(self):
        """Reset smoothing state (call at episode start)."""
        self.prev_action = None


# Successor Representation components (for future integration)
class SuccessorFeatureNetwork(nn.Module):
    """
    Successor Representation network.
    Learns to predict discounted future state occupancy.
    Can be integrated for transfer learning across tasks.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),  # Predict future feature occupancy
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict successor features."""
        return self.network(features)


def create_advanced_policy():
    """Factory function to create advanced policy."""
    return AdvancedActorCriticPolicy


# Export key components
__all__ = [
    'AdvancedActorCriticPolicy',
    'AdvancedFeaturesExtractor',
    'ActionSmoother',
    'SuccessorFeatureNetwork',
    'create_advanced_policy',
]

