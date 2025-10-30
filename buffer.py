import numpy as np
from typing import Optional
class PPOBuffer:
    def __init__(self, config, num_envs):
        self.config = config
        self.buffer_size = config.num_steps
        self.num_envs = num_envs

        obs_shape = (self.buffer_size, num_envs, config.window_size, 10)  # 10 features used
        act_shape = (self.buffer_size, num_envs, config.multi_asset_count)

        self.observations = np.zeros(obs_shape, dtype=np.float32)
        self.actions = np.zeros(act_shape, dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, num_envs), dtype=np.float32)
        self.extrinsic_rewards = np.zeros((self.buffer_size, num_envs), dtype=np.float32)
        self.intrinsic_rewards = np.zeros((self.buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size + 1, num_envs), dtype=np.bool_)

        self.h_in  = np.zeros((self.buffer_size, num_envs, config.lstm_hidden_size), dtype=np.float32)
        self.h_out = np.zeros((self.buffer_size, num_envs, config.lstm_hidden_size), dtype=np.float32)

        self.advantages = np.zeros((self.buffer_size, num_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, num_envs), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, value, log_prob, done, h_in, h_out, intrinsic_reward):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.extrinsic_rewards[self.ptr] = reward - intrinsic_reward
        self.intrinsic_rewards[self.ptr] = intrinsic_reward
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr + 1] = done

        if hasattr(h_in, "dim") and h_in.dim() == 3:
            h_in_py = h_in.squeeze(0).cpu().numpy()
        else:
            h_in_py = h_in.cpu().numpy()
        self.h_in[self.ptr] = h_in_py

        if hasattr(h_out, "dim") and h_out.dim() == 3:
            h_out_py = h_out.squeeze(0).cpu().numpy()
        else:
            h_out_py = h_out.cpu().numpy()
        self.h_out[self.ptr] = h_out_py

        self.ptr = (self.ptr + 1) % self.buffer_size

    def compute_gae_and_returns(self, next_value: np.ndarray, next_done: np.ndarray):
        self.dones[self.buffer_size] = next_done
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        curr_next_val = next_value
        for t in reversed(range(self.buffer_size)):
            non_terminal = 1.0 - self.dones[t + 1].astype(np.float32)
            delta = (self.rewards[t] + self.config.gamma * curr_next_val * non_terminal) - self.values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae
            curr_next_val = self.values[t]
        self.returns = self.advantages + self.values
        flat = self.advantages.flatten()
        mean = flat.mean() if flat.size else 0.0
        std = flat.std() + 1e-8
        self.advantages = (self.advantages - mean) / std
