# ppo_agent.py - PRODUCTION-READY PPO AGENT (LOGGER FIXED)
"""
Complete PPO agent with:
- IQN critic
- World model
- RND exploration
- Proper gradient handling
- Checkpointing
- Intrinsic reward normalization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from typing import Tuple, Dict, Optional
import os
import time
import logging
import tempfile

from model_core import PolicyIQNNet, WorldModel, RNDNet, init_hidden_state
from ppo_buffer import PPOBuffer

# FIXED: Add logger
logger = logging.getLogger(__name__)


class RunningMeanStd:
    """Running statistics for intrinsic reward normalization."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x) if isinstance(x, (list, np.ndarray)) else 1
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class PPOAgent:
    """Complete PPO agent with all auxiliary losses."""
    
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.device = config.device
        
        # Networks
        self.policy = PolicyIQNNet(env.observation_space, env.action_space, config).to(self.device)
        self.target_policy = PolicyIQNNet(env.observation_space, env.action_space, config).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        
        self.world_model = WorldModel(config).to(self.device)
        self.rnd_net = RNDNet(config).to(self.device)
        
        # Optimizers
        self.optimizer_actor = optim.Adam(
            self.policy.actor_head.parameters(),
            lr=config.learning_rate_actor,
            eps=1e-5
        )
        self.optimizer_critic = optim.Adam(
            list(self.policy.feature_rnn.parameters()) +
            list(self.policy.tau_fc.parameters()) +
            list(self.policy.critic_head.parameters()),
            lr=config.learning_rate_critic,
            eps=1e-5
        )
        self.optimizer_wm = optim.Adam(
            self.world_model.parameters(),
            lr=config.learning_rate_critic,
            eps=1e-5
        )
        self.optimizer_rnd = optim.Adam(
            self.rnd_net.predictor.parameters(),
            lr=config.rnd_predictor_lr,
            eps=1e-5
        )
        
        # Buffer
        self.buffer = PPOBuffer(config)
        
        # Internal state
        self.h_state = init_hidden_state(config, batch_size=1, device=self.device)
        
        # Intrinsic reward normalization
        self.rnd_rms = RunningMeanStd()
        
        # Training statistics
        self.total_updates = 0
        self.total_timesteps = 0
    
    def reset_recurrent_state(self):
        """Reset hidden state (e.g., at episode start)."""
        self.h_state = init_hidden_state(self.config, batch_size=1, device=self.device)
    
    def select_action(self, obs: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action using current policy.
        
        Args:
            obs: dict with 'features' and 'account'
            deterministic: use greedy action if True
        
        Returns:
            action: int
            log_prob: float
            value: float
        """
        with torch.no_grad():
            # Convert to tensors
            features_t = torch.from_numpy(obs["features"]).float().unsqueeze(0).to(self.device)
            account_t = torch.from_numpy(obs["account"]).float().unsqueeze(0).to(self.device)
            obs_t = {"features": features_t, "account": account_t}
            
            # Forward pass
            action_logits, _, h_out = self.policy(obs_t, self.h_state)
            dist = Categorical(logits=action_logits)
            
            # Sample or greedy
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            
            # Value estimate
            value, _ = self.policy.get_value_and_h(obs_t, self.h_state, k_quantiles=self.config.iqn_k_quantiles)
            
            # Update internal state
            self.h_state = h_out
        
        return action.item(), log_prob.item(), value.item()
    
    def compute_intrinsic_reward(self, next_obs_feat: np.ndarray) -> float:
        """
        Compute RND intrinsic reward.
        
        Args:
            next_obs_feat: (F,) next observation features
        
        Returns:
            intrinsic_reward: float
        """
        with torch.no_grad():
            next_feat_t = torch.from_numpy(next_obs_feat).float().unsqueeze(0).to(self.device)
            target_feat, pred_feat = self.rnd_net(next_feat_t)
            
            # Prediction error as novelty
            intrinsic_reward = F.mse_loss(pred_feat, target_feat, reduction='none').mean().item()
            
            # Normalize using running statistics
            self.rnd_rms.update([intrinsic_reward])
            normalized_reward = intrinsic_reward / (np.sqrt(self.rnd_rms.var) + 1e-8)
            
            # Scale and clip
            scaled_reward = normalized_reward * self.config.intrinsic_reward_scale
            clipped_reward = np.clip(scaled_reward, -self.config.intrinsic_reward_clip, 
                                     self.config.intrinsic_reward_clip)
        
        return float(clipped_reward)
    
    def _compute_gae_returns_batch(self, next_obs: Dict, done: bool) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute GAE with proper batch processing.
        
        Returns:
            advantages: (T,)
            returns: (T,)
            data: dict of tensors
        """
        # Get bootstrap value
        with torch.no_grad():
            features_t = torch.from_numpy(next_obs["features"]).float().unsqueeze(0).to(self.device)
            account_t = torch.from_numpy(next_obs["account"]).float().unsqueeze(0).to(self.device)
            next_obs_t = {"features": features_t, "account": account_t}
            
            next_value, _ = self.policy.get_value_and_h(next_obs_t, self.h_state, 
                                                        k_quantiles=self.config.iqn_k_quantiles)
            next_value = next_value.item() * (1.0 - float(done))
        
        # Compute GAE in buffer
        self.buffer.compute_gae_and_returns(next_value, done)
        
        # Get data
        data = self.buffer.get()
        
        # Normalize advantages
        advantages = data['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = data['returns']
        
        return advantages, returns, data
    
    def learn(self, next_obs: Dict, done: bool) -> Tuple[float, float, float, float, float]:
        """
        PPO learning step with all auxiliary losses.
        
        Returns:
            Tuple of (total_loss, policy_loss, critic_loss, wm_loss, rnd_loss)
        """
        # Compute advantages and returns
        advantages, returns, data = self._compute_gae_returns_batch(next_obs, done)
        
        # Move data to device
        obs_features = data['features'].to(self.device)
        obs_account = data['account'].to(self.device)
        actions = data['actions'].to(self.device)
        old_log_probs = data['log_probs'].to(self.device)
        rewards = data['rewards'].to(self.device)
        h_t_batch = data['h_t'].to(self.device)
        c_t_batch = data['c_t'].to(self.device)
        next_obs_features = data['next_features'].to(self.device)
        
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # Minibatch indices
        batch_indices = np.arange(self.config.rollout_steps)
        
        # Accumulate losses
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        critic_loss_sum = 0.0
        wm_loss_sum = 0.0
        rnd_loss_sum = 0.0
        num_updates = 0
        
        for epoch in range(self.config.ppo_epochs):
            np.random.shuffle(batch_indices)
            
            for start in range(0, self.config.rollout_steps, self.config.batch_size):
                end = min(start + self.config.batch_size, self.config.rollout_steps)
                idx = batch_indices[start:end]
                
                # Minibatch data
                mb_obs = {
                    "features": obs_features[idx],
                    "account": obs_account[idx]
                }
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]
                mb_next_feat = next_obs_features[idx]
                mb_rewards = rewards[idx]
                
                # Reconstruct hidden state (use first element's state for simplicity)
                mb_h_t = h_t_batch[idx[0]].unsqueeze(1).expand(-1, len(idx), -1).contiguous()
                mb_c_t = c_t_batch[idx[0]].unsqueeze(1).expand(-1, len(idx), -1).contiguous()
                mb_h_in = (mb_h_t, mb_c_t)
                
                # === POLICY LOSS ===
                action_logits, _, _ = self.policy(mb_obs, mb_h_in)
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy()
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 
                                   1.0 + self.config.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean() * self.config.entropy_beta
                
                # === IQN CRITIC LOSS ===
                B = len(idx)
                N = self.config.iqn_n_quantiles
                
                taus = torch.rand(B, N, device=self.device)
                _, current_quantiles, _ = self.policy(mb_obs, mb_h_in, taus)
                
                # Target quantiles
                with torch.no_grad():
                    target_quantiles_expanded = mb_returns.unsqueeze(-1).unsqueeze(1).expand(-1, N, -1)
                
                # Quantile Huber loss
                td_error = target_quantiles_expanded - current_quantiles.unsqueeze(-1)
                huber = F.huber_loss(
                    current_quantiles.unsqueeze(-1),
                    target_quantiles_expanded,
                    reduction='none',
                    delta=1.0
                )
                
                tau_weights = torch.abs(taus.unsqueeze(-1) - (td_error < 0).float())
                critic_loss = (tau_weights * huber).mean()
                
                # === WORLD MODEL LOSS ===
                wm_pred = self.world_model(mb_obs, mb_actions)
                
                # Target: shifted features + account + immediate reward
                wm_target_features_history = mb_obs["features"][:, 1:, :].flatten(start_dim=1)
                wm_target_features = torch.cat([wm_target_features_history, mb_next_feat], dim=1)
                
                wm_target = torch.cat([
                    wm_target_features,
                    mb_obs["account"],
                    mb_rewards.unsqueeze(-1)
                ], dim=-1)
                
                wm_loss = F.mse_loss(wm_pred, wm_target) * self.config.wm_loss_weight
                
                # === RND LOSS ===
                rnd_target, rnd_pred = self.rnd_net(mb_next_feat)
                rnd_loss = F.mse_loss(rnd_pred, rnd_target.detach()) * self.config.rnd_loss_weight
                
                # === TOTAL LOSS ===
                total_loss = policy_loss + critic_loss + entropy_loss + wm_loss + rnd_loss
                
                # === BACKPROPAGATION ===
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                self.optimizer_wm.zero_grad()
                self.optimizer_rnd.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.rnd_net.predictor.parameters(), self.config.max_grad_norm)
                
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                self.optimizer_wm.step()
                self.optimizer_rnd.step()
                
                # Accumulate metrics
                total_loss_sum += total_loss.item()
                policy_loss_sum += policy_loss.item()
                critic_loss_sum += critic_loss.item()
                wm_loss_sum += wm_loss.item()
                rnd_loss_sum += rnd_loss.item()
                num_updates += 1
        
        # Soft update target network
        self._soft_update(self.target_policy, self.policy, self.config.iqn_tau_soft_update)
        
        self.total_updates += num_updates
        
        # Return average losses
        return (
            total_loss_sum / num_updates,
            policy_loss_sum / num_updates,
            critic_loss_sum / num_updates,
            wm_loss_sum / num_updates,
            rnd_loss_sum / num_updates
        )
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )
    
    def save_checkpoint(self, path: str):
        """Save complete checkpoint with multiple fallback strategies."""
        checkpoint = {
            'policy': self.policy.state_dict(),
            'target_policy': self.target_policy.state_dict(),
            'world_model': self.world_model.state_dict(),
            'rnd_net': self.rnd_net.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'optimizer_wm': self.optimizer_wm.state_dict(),
            'optimizer_rnd': self.optimizer_rnd.state_dict(),
            'rnd_rms_mean': self.rnd_rms.mean,
            'rnd_rms_var': self.rnd_rms.var,
            'rnd_rms_count': self.rnd_rms.count,
            'total_updates': self.total_updates,
            'total_timesteps': self.total_timesteps,
            'config': vars(self.config)
        }
        
        # Try multiple locations with fallback
        save_locations = [
            path,
            f"/tmp/checkpoint_{int(time.time())}.pt",
            os.path.join(tempfile.gettempdir(), f"checkpoint_{int(time.time())}.pt"),
        ]
        
        for save_path in save_locations:
            try:
                dir_path = os.path.dirname(save_path)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                
                torch.save(checkpoint, save_path)
                logger.info(f"✅ Checkpoint saved: {save_path}")
                
                if save_path != path:
                    logger.info(f"   📋 Copy with: docker cp <container>:{save_path} ./checkpoint.pt")
                
                return save_path
                
            except Exception as e:
                logger.debug(f"Could not save to {save_path}: {e}")
                continue
        
        logger.error("❌ Failed to save checkpoint to any location")
        logger.info("   💾 Checkpoint is still in memory - keep container running")
        return None
    
    def load_checkpoint(self, path: str):
        """Load complete checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy'])
        self.target_policy.load_state_dict(checkpoint['target_policy'])
        self.world_model.load_state_dict(checkpoint['world_model'])
        self.rnd_net.load_state_dict(checkpoint['rnd_net'])
        
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        self.optimizer_wm.load_state_dict(checkpoint['optimizer_wm'])
        self.optimizer_rnd.load_state_dict(checkpoint['optimizer_rnd'])
        
        self.rnd_rms.mean = checkpoint['rnd_rms_mean']
        self.rnd_rms.var = checkpoint['rnd_rms_var']
        self.rnd_rms.count = checkpoint['rnd_rms_count']
        
        self.total_updates = checkpoint.get('total_updates', 0)
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        
        logger.info(f"✅ Checkpoint loaded: {path}")
    
    def save_model(self, path: str):
        """
        Save policy with multiple fallback strategies for Docker/Windows.
        
        Tries in order:
        1. Original path
        2. /tmp directory (always writable in Docker)
        3. Python's tempfile directory
        4. Emergency in-memory backup
        """
        model_data = {
            'policy': self.policy.state_dict(),
            'config': vars(self.config)
        }
        
        # Strategy 1: Try original path
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model_data, path)
            logger.info(f"✅ Model saved: {path}")
            return path
        except (PermissionError, RuntimeError, OSError) as e:
            logger.warning(f"⚠️ Could not save to {path}: {e}")
        
        # Strategy 2: Try /tmp directory with timestamp
        try:
            temp_path = f"/tmp/final_model_{int(time.time())}.pt"
            torch.save(model_data, temp_path)
            logger.info(f"✅ Model saved to temporary location: {temp_path}")
            logger.info(f"   📋 Copy with: docker cp <container>:{temp_path} ./final_model.pt")
            return temp_path
        except Exception as e:
            logger.warning(f"⚠️ Could not save to /tmp: {e}")
        
        # Strategy 3: Try Python's tempfile
        try:
            with tempfile.NamedTemporaryFile(
                mode='wb', 
                delete=False, 
                suffix='.pt',
                prefix='model_'
            ) as tmp_file:
                torch.save(model_data, tmp_file.name)
                logger.info(f"✅ Model saved to: {tmp_file.name}")
                logger.info(f"   📋 Copy with: docker cp <container>:{tmp_file.name} ./final_model.pt")
                return tmp_file.name
        except Exception as e:
            logger.warning(f"⚠️ Could not save to tempfile: {e}")
        
        # Strategy 4: Emergency - save to global variable
        logger.error("❌ Failed to save model to disk anywhere")
        logger.info("   💾 Model is still in memory - keep container running to extract it")
        logger.info("   🔧 Use: docker exec <container> python -c 'import torch; ...'")
        
        # Save to a global variable as last resort
        globals()['_emergency_model_backup'] = model_data
        logger.info("   💾 Model saved to emergency in-memory backup")
        
        return None
    
    def load_model(self, path: str):
        """Load policy only."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        logger.info(f"✅ Model loaded: {path}")