# ppo_buffer.py - PRODUCTION-READY ROLLOUT BUFFER (continued)
"""
Consolidated buffer implementation with:
- Proper memory management
- Intrinsic reward tracking
- Efficient GAE computation
- Vectorized operations
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


class PPOBuffer:
    """
    Unified rollout buffer for PPO with recurrent states.
    Handles both extrinsic and intrinsic rewards.
    """
    
    def __init__(self, config):
        self.config = config
        self.rollout_steps = config.rollout_steps
        self.window_size = config.window_size
        self.feature_dim = config.feature_dim
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        
        # Allocate storage
        self.memory = {
            "features": np.zeros((self.rollout_steps, self.window_size, self.feature_dim), dtype=np.float32),
            "account": np.zeros((self.rollout_steps, 3), dtype=np.float32),
            "next_features": np.zeros((self.rollout_steps, self.feature_dim), dtype=np.float32),
            "actions": np.zeros(self.rollout_steps, dtype=np.int64),
            "log_probs": np.zeros(self.rollout_steps, dtype=np.float32),
            "rewards": np.zeros(self.rollout_steps, dtype=np.float32),
            "extrinsic_rewards": np.zeros(self.rollout_steps, dtype=np.float32),
            "intrinsic_rewards": np.zeros(self.rollout_steps, dtype=np.float32),
            "values": np.zeros(self.rollout_steps, dtype=np.float32),
            "dones": np.zeros(self.rollout_steps, dtype=np.bool_),
            "h_t": np.zeros((self.rollout_steps, self.num_layers, self.hidden_size), dtype=np.float32),
            "c_t": np.zeros((self.rollout_steps, self.num_layers, self.hidden_size), dtype=np.float32),
        }
        
        # GAE buffers
        self.advantages = np.zeros(self.rollout_steps, dtype=np.float32)
        self.returns = np.zeros(self.rollout_steps, dtype=np.float32)
        
        self.ptr = 0
        self._is_full = False
    
    def add(self, obs: Dict, next_obs_feat: np.ndarray, action: int, 
            log_prob: float, reward: float, done: bool, h_in: Tuple,
            intrinsic_reward: float = 0.0, value: float = 0.0):
        """
        Add single transition to buffer.
        
        Args:
            obs: dict with 'features' and 'account'
            next_obs_feat: (F,) next state features
            action: int action
            log_prob: float log probability
            reward: float total reward
            done: bool terminal flag
            h_in: tuple (h_t, c_t) each (L, 1, H)
            intrinsic_reward: float RND reward
            value: float value estimate
        """
        if self.ptr >= self.rollout_steps:
            raise IndexError(f"Buffer overflow: {self.ptr}/{self.rollout_steps}")
        
        self.memory['features'][self.ptr] = obs["features"]
        self.memory['account'][self.ptr] = obs["account"]
        self.memory['next_features'][self.ptr] = next_obs_feat
        self.memory['actions'][self.ptr] = action
        self.memory['log_probs'][self.ptr] = log_prob
        self.memory['rewards'][self.ptr] = reward
        self.memory['extrinsic_rewards'][self.ptr] = reward - intrinsic_reward
        self.memory['intrinsic_rewards'][self.ptr] = intrinsic_reward
        self.memory['values'][self.ptr] = value
        self.memory['dones'][self.ptr] = done
        
        # Extract hidden states from tuple
        h_t, c_t = h_in
        self.memory['h_t'][self.ptr] = h_t.cpu().squeeze(1).numpy()  # (L, 1, H) -> (L, H)
        self.memory['c_t'][self.ptr] = c_t.cpu().squeeze(1).numpy()
        
        self.ptr += 1
        
        if self.ptr >= self.rollout_steps:
            self._is_full = True
    
    def compute_gae_and_returns(self, next_value: float, next_done: bool):
        """
        Compute GAE advantages and returns.
        
        Args:
            next_value: V(s_{T+1}) for bootstrapping
            next_done: terminal flag for next state
        """
        rewards = self.memory['rewards'][:self.ptr]
        values = self.memory['values'][:self.ptr]
        dones = self.memory['dones'][:self.ptr]
        
        advantages = np.zeros(self.ptr, dtype=np.float32)
        last_gae_lambda = 0.0
        
        # Backward GAE computation
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - float(next_done)
                next_val = next_value
            else:
                next_non_terminal = 1.0 - float(dones[t + 1])
                next_val = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae_lambda = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae_lambda
        
        returns = advantages + values
        
        # Store computed values
        self.advantages[:self.ptr] = advantages
        self.returns[:self.ptr] = returns
    
    def get(self) -> Dict[str, torch.Tensor]:
        """
        Convert stored data to torch tensors.
        
        Returns:
            Dictionary of tensors
        """
        if not self._is_full:
            # Partial buffer (edge case)
            data = {k: torch.from_numpy(v[:self.ptr].copy()) for k, v in self.memory.items()}
            data['advantages'] = torch.from_numpy(self.advantages[:self.ptr].copy())
            data['returns'] = torch.from_numpy(self.returns[:self.ptr].copy())
        else:
            data = {k: torch.from_numpy(v.copy()) for k, v in self.memory.items()}
            data['advantages'] = torch.from_numpy(self.advantages.copy())
            data['returns'] = torch.from_numpy(self.returns.copy())
        
        return data
    
    def get_statistics(self) -> Dict[str, float]:
        """Return buffer statistics for logging."""
        if self.ptr == 0:
            return {}
        
        return {
            'mean_reward': float(np.mean(self.memory['rewards'][:self.ptr])),
            'mean_extrinsic': float(np.mean(self.memory['extrinsic_rewards'][:self.ptr])),
            'mean_intrinsic': float(np.mean(self.memory['intrinsic_rewards'][:self.ptr])),
            'mean_value': float(np.mean(self.memory['values'][:self.ptr])),
            'mean_advantage': float(np.mean(self.advantages[:self.ptr])),
            'std_advantage': float(np.std(self.advantages[:self.ptr])),
        }
    
    def reset(self):
        """Reset buffer pointer and flags."""
        self.ptr = 0
        self._is_full = False
        # No need to zero out arrays - they'll be overwritten
    
    def is_full(self) -> bool:
        """Check if buffer is ready for training."""
        return self._is_full
    
    def __len__(self) -> int:
        """Return number of stored transitions."""
        return self.ptr