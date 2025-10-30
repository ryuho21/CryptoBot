import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple

# Utility function for soft target update (left for potential future use)
def soft_update(target, source, tau):
    """Soft update model parameters."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

class PolicyNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.W = config.window_size
        
        # --- FIXED: Feature dimension to 10 ---
        self.F_in = getattr(config, 'feature_dim', 10) 
        
        # --- FIXED: Corrected Attribute Name (Previous AttributeError) ---
        self.H = getattr(config, 'lstm_hidden_size', 128)
        
        # Action dimension (2 for continuous trading: e.g., position size, risk exposure)
        self.A = getattr(config, 'multi_asset_count', 2) 
        
        self.rnn_layers = getattr(config, 'rnn_layers', 1)

        # --- Feature Projection: (B*W, F_in) -> (B*W, H) ---
        # Projects each time step's feature vector (F_in) into the hidden space (H)
        self.feature_proj = nn.Sequential(
            nn.Linear(self.F_in, self.H), 
            nn.ReLU()
        )

        # --- Recurrent Core (GRU or LSTM) ---
        self.rnn_type = getattr(config, 'rnn_type', 'GRU').upper()
        
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.H,
                hidden_size=self.H,
                num_layers=self.rnn_layers,
                batch_first=True # (B, T, *)
            )
        else: # GRU (Default)
            self.rnn = nn.GRU(
                input_size=self.H,
                hidden_size=self.H,
                num_layers=self.rnn_layers,
                batch_first=True # (B, T, *)
            )
        
        # --- Actor Head (Gaussian Policy) ---
        self.actor_head = nn.Sequential(
            nn.Linear(self.H, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * self.A) # Outputs mean and log_std for 2 actions
        )
        self.log_std_min = -20
        self.log_std_max = 2

        # --- Critic Head (Standard V(s) for PPO) ---
        self.critic_head = nn.Sequential(
            nn.Linear(self.H, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs V(s) scalar
        )
        
    def _forward_rnn(self, x: torch.Tensor, h_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handles the RNN forward pass for GRU and LSTM."""
        # x shape: (B, W, H)
        if self.rnn_type == 'LSTM':
            if isinstance(h_in, torch.Tensor): 
                 # Convert single tensor (L, B, H) to LSTM tuple state (h_t, c_t)
                 h_in = (h_in, torch.zeros_like(h_in))
            rnn_out, h_out = self.rnn(x, h_in)
        else: # GRU
            if isinstance(h_in, tuple):
                h_in = h_in[0] # Take only h_t from an LSTM tuple if mistakenly passed
            rnn_out, h_out = self.rnn(x, h_in)
        # rnn_out shape: (B, W, H)
        return rnn_out, h_out

    def forward(self, obs: torch.Tensor, h_in):
        """Standard forward pass for PPO. Returns action distribution, value, and new recurrent state."""
        # obs shape: (B, W, F_in), h_in shape: (L, B, H) or tuple for LSTM
        B, W, F = obs.shape
        
        # 1. Feature Projection (per time step): (B*W, F_in) -> (B*W, H)
        x = self.feature_proj(obs.reshape(B * W, F)).reshape(B, W, -1) # (B, W, H)
        
        # 2. Recurrent Core
        rnn_out, h_out = self._forward_rnn(x, h_in)
            
        # Use the last output from the RNN
        last_rnn_out = rnn_out[:, -1, :] # (B, H)
        
        # 3. Actor Head (Gaussian Policy)
        actor_output = self.actor_head(last_rnn_out)
        mean, log_std = actor_output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        action_dist = Normal(mean, std)
        
        # 4. Critic Head (Standard V(s))
        values = self.critic_head(last_rnn_out) # (B, 1)

        # Returns: (Normal distribution object, values tensor (B, 1), hidden state)
        return action_dist, values, h_out

    def get_value(self, obs: torch.Tensor, h_in: torch.Tensor):
        """Get value only (used for GAE bootstrap). Returns value and new recurrent state."""
        # obs shape: (B, W, F_in)
        B, W, F = obs.shape
        
        # 1. Feature Projection
        x = self.feature_proj(obs.reshape(B * W, F)).reshape(B, W, -1)
        
        # 2. Recurrent Core
        rnn_out, h_out = self._forward_rnn(x, h_in)
            
        # Use the last output from the RNN
        last_rnn_out = rnn_out[:, -1, :]
        
        # 3. Critic Head (Standard V(s))
        values = self.critic_head(last_rnn_out) # (B, 1)

        # Returns: (values tensor (B, 1), hidden state)
        return values, h_out # <-- ONLY returns 2 values