# trading_env.py - PRODUCTION-READY TRADING ENVIRONMENT
"""
Complete trading environment with:
- Proper observation/action spaces
- Reward shaping
- Stop-loss/take-profit
- Curriculum learning integration
- Performance metrics
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional
import logging

from config import Config
from data_pipeline import ExchangeDataSource
from exchange_manager import ExchangeManager, RiskManager
from curriculum_scheduler import CurriculumScheduler

logger = logging.getLogger(__name__)


class TradingBotEnv(gym.Env):
    """
    Production-ready trading environment.
    
    Observation space:
        Dict with:
        - features: (W, F) market features
        - account: (3,) [net_worth, position, volatility]
    
    Action space:
        Discrete(3): [HOLD, BUY, SELL]
    """
    
    metadata = {'render_modes': ['human', 'jsonl']}
    
    def __init__(self, data_source: ExchangeDataSource, 
                 exchange_manager: ExchangeManager, config: Config):
        super(TradingBotEnv, self).__init__()
        
        self.data_source = data_source
        self.exchange_manager = exchange_manager
        self.risk_manager = exchange_manager.risk_manager
        self.config = config
        
        # Curriculum scheduler
        self.curriculum = CurriculumScheduler(config)
        
        # Observation space
        self.observation_space = spaces.Dict({
            "features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=config.observation_shape,
                dtype=np.float32
            ),
            "account": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float32
            ),
        })
        
        # Action space
        self.action_space = spaces.Discrete(config.action_dim)
        
        # Environment state
        self.initial_balance = config.initial_balance
        self.commission = config.commission_per_trade
        self.max_steps = len(self.data_source.data_store) - config.window_size - 1
        
        self.current_step = 0
        self.entry_price = 0.0
        self.trade_count = 0
        self.episode_reward = 0.0
        
        # Performance tracking
        self.equity_curve = []
        self.trade_log = []
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Reset financial state
        self.risk_manager.net_worth = self.initial_balance
        self.risk_manager.max_net_worth = self.initial_balance
        self.risk_manager.max_drawdown = 0.0
        self.exchange_manager.current_position = 0.0
        self.exchange_manager.position_entry_price = 0.0
        
        # Reset episode state
        self.current_step = self.config.window_size
        self.entry_price = 0.0
        self.trade_count = 0
        self.episode_reward = 0.0
        self.equity_curve = [self.initial_balance]
        self.trade_log = []
        
        # Update curriculum
        self.curriculum.update({"episode_reward": 0})
        
        observation = self._get_obs(self.current_step)
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        terminated = False
        truncated = False
        reward = 0.0
        
        # Check episode termination
        if self.current_step >= self.max_steps:
            truncated = True
        
        current_price = self.data_source.get_market_price(self.current_step)
        
        if not (terminated or truncated):
            # Update unrealized PnL
            reward += self._update_unrealized_pnl(current_price)
            
            # Execute trading action
            trade_reward = self._execute_trade_action(action, current_price)
            reward += trade_reward
            
            # Check stop-loss/take-profit
            sl_tp_reward = self._check_stop_loss_take_profit(current_price)
            reward += sl_tp_reward
            
            # Check risk limits
            if self.risk_manager.net_worth <= 0 or self.risk_manager.circuit_breaker_tripped:
                terminated = True
                reward -= 100.0  # Penalty for blowing up
            
            # Update equity curve
            self.equity_curve.append(self.risk_manager.net_worth)
            
            observation = self._get_obs(self.current_step)
        else:
            # Episode end: close all positions
            final_reward = self._close_final_position(current_price)
            reward += final_reward
            observation = self._get_obs(self.current_step - 1)
        
        # Update curriculum with episode reward
        self.episode_reward += reward
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self, index: int) -> Dict[str, np.ndarray]:
        """
        Get observation at given step.
        
        Returns:
            Dict with 'features' and 'account'
        """
        # Market features
        features = self.data_source.get_latest_observation()
        
        # Account state
        norm_net_worth = self.risk_manager.net_worth / self.initial_balance
        
        # Normalize position by max leverage
        max_position_value = self.initial_balance * self.config.max_position_leverage
        current_price = self.data_source.get_market_price(min(index, len(self.data_source.data_store) - 1))
        position_value = self.exchange_manager.current_position * current_price
        norm_position = position_value / max_position_value
        
        # Volatility factor from curriculum
        volatility_factor = self.curriculum.get_volatility_factor()
        
        account_state = np.array([
            norm_net_worth,
            norm_position,
            volatility_factor
        ], dtype=np.float32)
        
        return {
            "features": features,
            "account": account_state
        }
    
    def _update_unrealized_pnl(self, price: float) -> float:
        """
        Calculate unrealized PnL change as reward.
        
        Returns:
            reward: float
        """
        position = self.exchange_manager.current_position
        
        if np.isclose(position, 0.0) or np.isclose(self.entry_price, 0.0):
            return 0.0
        
        # PnL calculation
        if position > 0:  # Long
            pnl = (price - self.entry_price) * position
        else:  # Short
            pnl = (self.entry_price - price) * abs(position)
        
        # Reward as percentage of initial balance (normalized)
        reward = np.tanh(pnl / self.initial_balance) * 0.1
        
        return reward
    
    def _execute_trade_action(self, action: int, price: float) -> float:
        """
        Execute trading action and return reward.
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            price: current market price
        
        Returns:
            reward: float
        """
        action_type = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}[action]
        position = self.exchange_manager.current_position
        
        if action_type == 'HOLD':
            return 0.0
        
        # Calculate position size
        position_value = position * price
        size_usd = self.risk_manager.calculate_position_size_usd(
            price,
            position_value,
            self.config.kelly_criterion_fraction
        )
        
        # Check minimum size
        if size_usd < 10.0:
            return 0.0
        
        # Execute order
        success, result = self.exchange_manager.execute_order(action_type, price, size_usd)
        
        if not success:
            return -0.01  # Small penalty for failed orders
        
        # Calculate reward based on position change
        new_position = self.exchange_manager.current_position
        reward = 0.0
        
        # Position closed (realized PnL)
        if np.isclose(new_position, 0.0) and not np.isclose(position, 0.0):
            # Calculate realized PnL
            if position > 0:  # Closing long
                pnl = (price - self.entry_price) * abs(position)
            else:  # Closing short
                pnl = (self.entry_price - price) * abs(position)
            
            # Commission
            commission = abs(position) * price * self.commission
            net_pnl = pnl - commission
            
            # Update net worth
            self.risk_manager.net_worth += net_pnl
            self.risk_manager.update_metrics(self.risk_manager.net_worth)
            
            # Reward proportional to PnL
            reward = (net_pnl / self.initial_balance) * 50.0
            
            # Log trade
            self.trade_log.append({
                'step': self.current_step,
                'action': 'CLOSE',
                'price': price,
                'pnl': net_pnl,
                'net_worth': self.risk_manager.net_worth
            })
            
            self.entry_price = 0.0
            self.trade_count += 1
        
        # Position opened
        elif np.isclose(position, 0.0) and not np.isclose(new_position, 0.0):
            self.entry_price = price
            
            # Commission cost
            commission = abs(new_position) * price * self.commission
            self.risk_manager.net_worth -= commission
            
            # Small penalty for opening (encourages selective trading)
            reward = -self.commission * 5.0
            
            # Log trade
            self.trade_log.append({
                'step': self.current_step,
                'action': action_type,
                'price': price,
                'size': new_position,
                'net_worth': self.risk_manager.net_worth
            })
        
        # Position adjusted (partial close or add)
        else:
            commission = abs(new_position - position) * price * self.commission
            self.risk_manager.net_worth -= commission
            reward = -self.commission * 2.0
        
        return reward
    
    def _check_stop_loss_take_profit(self, price: float) -> float:
        """
        Check and execute stop-loss or take-profit.
        
        Returns:
            reward: float
        """
        position = self.exchange_manager.current_position
        
        if np.isclose(position, 0.0) or np.isclose(self.entry_price, 0.0):
            return 0.0
        
        reward = 0.0
        
        if position > 0:  # Long position
            pnl_pct = (price - self.entry_price) / self.entry_price
            
            # Stop loss
            if pnl_pct <= -self.config.stop_loss_pct:
                logger.info(f"🛑 Stop-loss triggered: {pnl_pct:.2%}")
                success, _ = self.exchange_manager.execute_order('SELL', price, position * price)
                if success:
                    reward = -10.0  # Penalty for stop-loss
            
            # Take profit
            elif pnl_pct >= self.config.take_profit_pct:
                logger.info(f"💰 Take-profit triggered: {pnl_pct:.2%}")
                success, _ = self.exchange_manager.execute_order('SELL', price, position * price)
                if success:
                    reward = 5.0  # Reward for take-profit
        
        else:  # Short position
            pnl_pct = (self.entry_price - price) / self.entry_price
            
            # Stop loss
            if pnl_pct <= -self.config.stop_loss_pct:
                logger.info(f"🛑 Stop-loss triggered: {pnl_pct:.2%}")
                success, _ = self.exchange_manager.execute_order('BUY', price, abs(position) * price)
                if success:
                    reward = -10.0
            
            # Take profit
            elif pnl_pct >= self.config.take_profit_pct:
                logger.info(f"💰 Take-profit triggered: {pnl_pct:.2%}")
                success, _ = self.exchange_manager.execute_order('BUY', price, abs(position) * price)
                if success:
                    reward = 5.0
        
        return reward
    
    def _close_final_position(self, price: float) -> float:
        """
        Close all positions at episode end.
        
        Returns:
            reward: float
        """
        position = self.exchange_manager.current_position
        
        if np.isclose(position, 0.0):
            # No position to close
            final_return = (self.risk_manager.net_worth - self.initial_balance) / self.initial_balance
            return final_return * 100.0
        
        # Calculate final PnL
        if position > 0:
            pnl = (price - self.entry_price) * position
        else:
            pnl = (self.entry_price - price) * abs(position)
        
        commission = abs(position) * price * self.commission
        net_pnl = pnl - commission
        
        self.risk_manager.net_worth += net_pnl
        self.exchange_manager.current_position = 0.0
        
        # Final return as reward
        final_return = (self.risk_manager.net_worth - self.initial_balance) / self.initial_balance
        
        return final_return * 100.0
    
    def _get_info(self) -> Dict:
        """Get environment info."""
        current_price = self.data_source.get_market_price(
            min(self.current_step, len(self.data_source.data_store) - 1)
        )
        
        return {
            "net_worth": self.risk_manager.net_worth,
            "position": self.exchange_manager.current_position,
            "current_price": current_price,
            "max_drawdown": self.risk_manager.max_drawdown,
            "circuit_tripped": self.risk_manager.circuit_breaker_tripped,
            "trade_count": self.trade_count,
            "episode_reward": self.episode_reward,
            "equity_curve": self.equity_curve[-100:],  # Last 100 steps
            "volatility_factor": self.curriculum.get_volatility_factor()
        }
    
    def render(self, mode: str = 'human'):
        """Render environment state."""
        if mode == 'human':
            info = self._get_info()
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Net Worth: ${info['net_worth']:,.2f}")
            print(f"Position: {info['position']:.4f}")
            print(f"Price: ${info['current_price']:.2f}")
            print(f"Max DD: {info['max_drawdown']:.2%}")
            print(f"Trades: {info['trade_count']}")
            print(f"Episode Reward: {info['episode_reward']:.2f}")
            print(f"{'='*60}\n")
    
    def get_episode_statistics(self) -> Dict:
        """Calculate episode performance statistics."""
        equity = np.array(self.equity_curve)
        
        # Returns
        returns = np.diff(equity) / equity[:-1]
        
        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino = 0.0
        
        # Win rate
        winning_trades = sum(1 for t in self.trade_log if t.get('pnl', 0) > 0)
        total_trades = len([t for t in self.trade_log if 'pnl' in t])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': self.risk_manager.max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_net_worth': equity[-1]
        }
