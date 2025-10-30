# model_core.py - PRODUCTION-READY NEURAL NETWORKS
"""
Complete implementation of:
- PolicyIQNNet (Actor-Critic with IQN distributional RL)
- WorldModel (Dynamics prediction)
- RNDNet (Intrinsic motivation)
- Proper state management
- Gradient stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Optional


class PolicyIQNNet(nn.Module):
    """
    Combined Actor-Critic with IQN (Implicit Quantile Networks).
    Supports both GRU and LSTM with unified tuple state format.
    """
    def __init__(self, observation_space, action_space, config):
        super(PolicyIQNNet, self).__init__()
        
        self.config = config
        num_actions = action_space.n
        num_features = config.feature_dim
        
        # RNN configuration
        self.rnn_type = getattr(config, 'rnn_type', 'LSTM').upper()
        self.hidden_size = getattr(config, 'hidden_size', 256)
        self.num_layers = getattr(config, 'num_layers', 2)
        
        # Recurrent Feature Extractor
        if self.rnn_type == 'LSTM':
            self.feature_rnn = nn.LSTM(
                input_size=num_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=0.1 if self.num_layers > 1 else 0.0
            )
        else:  # GRU
            self.feature_rnn = nn.GRU(
                input_size=num_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=0.1 if self.num_layers > 1 else 0.0
            )
        
        # Combined input: RNN output + account state
        account_dim = observation_space["account"].shape[0]
        combined_input_size = self.hidden_size + account_dim
        
        # === ACTOR HEAD ===
        self.actor_head = nn.Sequential(
            nn.Linear(combined_input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, num_actions)
        )
        
        # === CRITIC HEAD (IQN) ===
        self.iqn_embedding_dim = getattr(config, 'iqn_hidden_size', 64)
        self.tau_fc = nn.Linear(self.iqn_embedding_dim, combined_input_size)
        
        self.critic_head = nn.Sequential(
            nn.Linear(combined_input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Orthogonal initialization for stability."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.zero_()
    
    def _normalize_hidden_state(self, h_in):
        """Convert to LSTM tuple format (h_t, c_t) for consistency."""
        if self.rnn_type == 'LSTM':
            if not isinstance(h_in, tuple):
                h_in = (h_in, torch.zeros_like(h_in))
            return h_in
        else:  # GRU
            if isinstance(h_in, tuple):
                return h_in[0]
            return h_in
    
    def _denormalize_hidden_state(self, h_out):
        """Convert back to tuple format."""
        if self.rnn_type == 'LSTM':
            return h_out
        else:  # GRU
            return (h_out, torch.zeros_like(h_out))
    
    def _get_rnn_output(self, features, h_in):
        """
        Process sequence through RNN.
        
        Args:
            features: (B, W, F)
            h_in: tuple (h_t, c_t) each (L, B, H)
        
        Returns:
            last_output: (B, H)
            h_out: tuple (h_t, c_t)
        """
        h_in_normalized = self._normalize_hidden_state(h_in)
        rnn_out, h_out_raw = self.feature_rnn(features, h_in_normalized)
        last_output = rnn_out[:, -1, :]
        h_out = self._denormalize_hidden_state(h_out_raw)
        return last_output, h_out
    
    def forward(self, obs, h_in, taus=None):
        """
        Forward pass.
        
        Args:
            obs: dict with 'features' (B, W, F) and 'account' (B, A)
            h_in: tuple (h_t, c_t)
            taus: (B, N) quantiles for IQN
        
        Returns:
            action_logits: (B, num_actions)
            q_quantiles: (B, N) or None
            h_out: tuple (h_t, c_t)
        """
        features = obs["features"]
        account = obs["account"]
        
        # RNN processing
        last_rnn_output, h_out = self._get_rnn_output(features, h_in)
        combined = torch.cat((last_rnn_output, account), dim=-1)
        
        # Actor
        action_logits = self.actor_head(combined)
        
        # Critic (IQN)
        q_quantiles = None
        if taus is not None:
            B, N = taus.shape
            
            # Cosine embedding
            tau_range = torch.arange(1, self.iqn_embedding_dim + 1,
                                    device=taus.device, dtype=torch.float32)
            tau_cos_emb = torch.cos(taus.unsqueeze(-1) * np.pi * tau_range)
            
            # Project and combine
            tau_emb = F.relu(self.tau_fc(tau_cos_emb))
            combined_expanded = combined.unsqueeze(1).expand(-1, N, -1)
            iqn_input = combined_expanded * tau_emb
            
            q_quantiles = self.critic_head(iqn_input).squeeze(-1)
        
        return action_logits, q_quantiles, h_out
    
    def get_value_and_h(self, obs, h_in, k_quantiles=32):
        """
        Estimate V(s) by averaging quantiles.
        
        Returns:
            value: (B, 1)
            h_out: tuple (h_t, c_t)
        """
        B = obs["features"].shape[0]
        device = obs["features"].device
        
        taus = torch.rand(B, k_quantiles, device=device)
        _, q_quantiles, h_out = self.forward(obs, h_in, taus)
        value = q_quantiles.mean(dim=-1, keepdim=True)
        
        return value, h_out


class WorldModel(nn.Module):
    """
    Predict next state and reward for model-based planning.
    
    Input: [features (F*W), account (3), action one-hot (num_actions)]
    Output: [next_features (F*W), next_account (3), reward (1)]
    """
    def __init__(self, config, num_actions=3):
        super(WorldModel, self).__init__()
        
        num_features = config.feature_dim
        window_size = config.window_size
        account_dim = 3
        
        input_size = num_features * window_size + account_dim + num_actions
        output_size = num_features * window_size + account_dim + 1
        
        hidden_size = getattr(config, 'hidden_size', 256)
        
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )
        
        self.feature_dim = num_features
        self.window_size = window_size
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, obs, action):
        """
        Args:
            obs: dict with 'features' (B, W, F) and 'account' (B, 3)
            action: (B,) action indices
        
        Returns:
            prediction: (B, F*W + 3 + 1)
        """
        features = obs["features"].flatten(start_dim=1)
        account = obs["account"]
        
        num_actions = 3
        action_one_hot = F.one_hot(action, num_classes=num_actions).float()
        
        wm_input = torch.cat((features, account, action_one_hot), dim=-1)
        prediction = self.model(wm_input)
        
        return prediction


class RNDNet(nn.Module):
    """
    Random Network Distillation for exploration.
    
    Input: Single feature vector (B, F)
    Output: Target and predictor embeddings (B, H)
    """
    def __init__(self, config):
        super(RNDNet, self).__init__()
        
        input_size = config.feature_dim
        hidden_size = getattr(config, 'hidden_size', 256)
        
        # Target (frozen)
        self.target = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        # Predictor (trainable)
        self.predictor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        # Freeze target
        for param in self.target.parameters():
            param.requires_grad = False
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, next_obs_features):
        """
        Args:
            next_obs_features: (B, F)
        
        Returns:
            target_output: (B, H/4)
            predictor_output: (B, H/4)
        """
        with torch.no_grad():
            target_output = self.target(next_obs_features)
        predictor_output = self.predictor(next_obs_features)
        return target_output, predictor_output


def init_hidden_state(config, batch_size, device):
    """Initialize recurrent hidden state in tuple format."""
    hidden_size = getattr(config, 'hidden_size', 256)
    num_layers = getattr(config, 'num_layers', 2)
    
    h_t = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    c_t = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    
    return (h_t, c_t)


def batch_init_hidden_states(config, batch_size, device):
    """Batch initialization for parallel environments."""
    return init_hidden_state(config, batch_size, device)