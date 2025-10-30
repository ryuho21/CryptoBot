# curriculum_scheduler.py - PRODUCTION-READY CURRICULUM SCHEDULER
"""
Adaptive curriculum learning that adjusts:
- Market volatility
- Spread and slippage
- Episode difficulty
Based on agent performance
"""

import os
import json
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """
    Adaptive curriculum scheduler for progressive difficulty.
    
    Adjusts environment parameters based on:
    - Recent performance (rewards, Sharpe ratio)
    - Win rate
    - Drawdown control
    """
    
    def __init__(self, config):
        self.config = config
        
        # Curriculum parameters
        self.min_vol = getattr(config, "initial_volatility_factor", 0.5)
        self.max_vol = getattr(config, "final_volatility_factor", 1.5)
        self.window = getattr(config, "curriculum_window", 100)
        self.delta_step = getattr(config, "curriculum_delta", 0.05)
        
        # State
        self.episode_rewards = []
        self.episode_sharpes = []
        self.episode_win_rates = []
        self.current_vol_mult = 1.0
        self.current_difficulty = 0.5  # 0 to 1 scale
        
        # Persistence
        self.history_file = os.path.join(config.runs_dir, "curriculum.json")
        self._load()
    
    def _load(self):
        """Load previous curriculum state."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as f:
                    data = json.load(f)
                self.current_vol_mult = float(data.get("current_vol_mult", 1.0))
                self.current_difficulty = float(data.get("current_difficulty", 0.5))
                logger.info(f"✅ Curriculum loaded: vol={self.current_vol_mult:.2f}, "
                          f"difficulty={self.current_difficulty:.2f}")
        except Exception as e:
            logger.warning(f"Could not load curriculum: {e}")
    
    def _save(self):
        """Save current curriculum state."""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, "w") as f:
                json.dump({
                    "current_vol_mult": self.current_vol_mult,
                    "current_difficulty": self.current_difficulty,
                    "episode_count": len(self.episode_rewards)
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save curriculum: {e}")
    
    def record_episode(self, episode_reward: float, sharpe: Optional[float] = None,
                      win_rate: Optional[float] = None):
        """Record episode metrics."""
        self.episode_rewards.append(episode_reward)
        
        if sharpe is not None:
            self.episode_sharpes.append(sharpe)
        
        if win_rate is not None:
            self.episode_win_rates.append(win_rate)
        
        # Keep window size
        if len(self.episode_rewards) > self.window:
            self.episode_rewards.pop(0)
        if len(self.episode_sharpes) > self.window:
            self.episode_sharpes.pop(0)
        if len(self.episode_win_rates) > self.window:
            self.episode_win_rates.pop(0)
    
    def get_mean_reward(self) -> float:
        """Get mean reward over window."""
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
    
    def get_mean_sharpe(self) -> float:
        """Get mean Sharpe ratio over window."""
        return float(np.mean(self.episode_sharpes)) if self.episode_sharpes else 0.0
    
    def get_mean_win_rate(self) -> float:
        """Get mean win rate over window."""
        return float(np.mean(self.episode_win_rates)) if self.episode_win_rates else 0.5
    
    def update(self, metrics: Dict) -> Dict:
        """
        Update curriculum based on performance metrics.
        
        Args:
            metrics: dict with 'episode_reward', 'sharpe', 'win_rate'
        
        Returns:
            Dict with updated curriculum parameters
        """
        # Record metrics
        ep_reward = metrics.get("episode_reward", 0.0)
        sharpe = metrics.get("sharpe", None)
        win_rate = metrics.get("win_rate", None)
        
        self.record_episode(ep_reward, sharpe, win_rate)
        
        # Not enough data yet
        if len(self.episode_rewards) < 10:
            return self._get_status()
        
        # Calculate performance indicators
        mean_reward = self.get_mean_reward()
        mean_sharpe = self.get_mean_sharpe()
        mean_win_rate = self.get_mean_win_rate()
        
        # Adaptive rules
        performance_score = 0.0
        
        # Reward criterion
        if mean_reward > 50:
            performance_score += 0.4
        elif mean_reward > 0:
            performance_score += 0.2
        
        # Sharpe criterion
        if mean_sharpe > 1.5:
            performance_score += 0.3
        elif mean_sharpe > 0.5:
            performance_score += 0.15
        
        # Win rate criterion
        if mean_win_rate > 0.6:
            performance_score += 0.3
        elif mean_win_rate > 0.5:
            performance_score += 0.15
        
        # Adjust difficulty
        if performance_score >= 0.7:
            # Good performance: increase difficulty
            self.current_difficulty = min(1.0, self.current_difficulty + self.delta_step)
            self.current_vol_mult = min(self.max_vol, self.current_vol_mult + self.delta_step)
            logger.info(f"📈 Curriculum increased: difficulty={self.current_difficulty:.2f}")
        
        elif performance_score < 0.3:
            # Poor performance: decrease difficulty
            self.current_difficulty = max(0.0, self.current_difficulty - self.delta_step)
            self.current_vol_mult = max(self.min_vol, self.current_vol_mult - self.delta_step)
            logger.info(f"📉 Curriculum decreased: difficulty={self.current_difficulty:.2f}")
        
        self._save()
        return self._get_status()
    
    def get_volatility_factor(self) -> float:
        """Return current volatility multiplier."""
        return self.current_vol_mult
    
    def get_difficulty(self) -> float:
        """Return current difficulty level (0 to 1)."""
        return self.current_difficulty
    
    def apply(self, target):
        """
        Apply curriculum parameters to target environment.
        
        Args:
            target: Environment or data manager to apply curriculum to
        """
        mult = self.current_vol_mult
        
        try:
            # Apply to config
            if hasattr(target, "config"):
                base_spread = 0.0005
                base_slip = 0.001
                target.config.spread_pct = float(base_spread * mult)
                target.config.slippage_pct = float(base_slip * mult)
                logger.debug(f"Applied curriculum: spread={target.config.spread_pct:.5f}, "
                           f"slippage={target.config.slippage_pct:.5f}")
            
            # Apply to volatility setter
            elif hasattr(target, "set_volatility"):
                target.set_volatility(mult)
        
        except Exception as e:
            logger.warning(f"Could not apply curriculum: {e}")
    
    def _get_status(self) -> Dict:
        """Get current curriculum status."""
        return {
            'current_vol_mult': self.current_vol_mult,
            'current_difficulty': self.current_difficulty,
            'mean_reward': self.get_mean_reward(),
            'mean_sharpe': self.get_mean_sharpe(),
            'mean_win_rate': self.get_mean_win_rate(),
            'episodes_recorded': len(self.episode_rewards)
        }
    
    def reset(self):
        """Reset curriculum to initial state."""
        self.current_vol_mult = 1.0
        self.current_difficulty = 0.5
        self.episode_rewards.clear()
        self.episode_sharpes.clear()
        self.episode_win_rates.clear()
        self._save()
        logger.info("🔄 Curriculum reset to initial state")