import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
import logging

from config import Config
from data_pipeline import ExchangeDataSource
from exchange_manager import ExchangeManager, RiskManager
from curriculum_scheduler import CurriculumScheduler

logger = logging.getLogger(__name__)


class TradingBotEnv(gym.Env):
    """
    Production-ready trading environment with IMPROVED REWARDS.
    
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
        
        # NEW: Improved reward tracking
        self.recent_returns = deque(maxlen=100)
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.winning_trades = 0
        self.time_in_position = 0
        self.last_portfolio_value = self.initial_balance
        
        # Reward component weights (can tune these)
        self.reward_weights = {
            'pnl': 1.0,              # Primary: profit/loss
            'opportunity_cost': 0.2, # Penalty: missing trends
            'holding_bonus': 0.15,   # Bonus: riding winners
            'action_penalty': 1.0,   # Cost: trading fees
        }
        
        logger.info("✅ TradingBotEnv initialized with IMPROVED REWARDS")
    
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
        
        # NEW: Reset improved tracking
        self.recent_returns.clear()
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.winning_trades = 0
        self.time_in_position = 0
        self.last_portfolio_value = self.initial_balance
        
        # Update curriculum
        self.curriculum.update({"episode_reward": 0})
        
        observation = self._get_obs(self.current_step)
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one environment step with IMPROVED REWARD CALCULATION.
        
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
            # Calculate IMPROVED reward
            reward = self._calculate_improved_reward(action, current_price)
            
            # Check stop-loss/take-profit
            sl_tp_reward = self._check_stop_loss_take_profit(current_price)
            reward += sl_tp_reward
            
            # Check risk limits
            if self.risk_manager.net_worth <= 0 or self.risk_manager.circuit_breaker_tripped:
                terminated = True
                reward -= 100.0  # Penalty for blowing up
            
            # Update equity curve and returns
            self.equity_curve.append(self.risk_manager.net_worth)
            
            # Track returns
            portfolio_return = (self.risk_manager.net_worth - self.last_portfolio_value) / self.last_portfolio_value
            self.recent_returns.append(portfolio_return)
            self.last_portfolio_value = self.risk_manager.net_worth
            
            # Track time in position
            if abs(self.exchange_manager.current_position) > 1e-6:
                self.time_in_position += 1
            else:
                self.time_in_position = 0
            
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
    
    def _calculate_improved_reward(self, action: int, price: float) -> float:
        """
        IMPROVED REWARD FUNCTION with multiple components.
        
        Components:
        1. PnL reward (primary)
        2. Opportunity cost (penalty for missing trends)
        3. Position holding bonus (let winners run)
        4. Action penalty (trading fees)
        
        Returns:
            total_reward: float
        """
        total_reward = 0.0
        
        # 1. PNL REWARD (Primary)
        pnl_reward = self._calculate_pnl_reward(action, price)
        total_reward += pnl_reward * self.reward_weights['pnl']
        
        # 2. OPPORTUNITY COST (Prevents always holding)
        opp_cost = self._calculate_opportunity_cost(action)
        total_reward += opp_cost * self.reward_weights['opportunity_cost']
        
        # 3. POSITION HOLDING BONUS (Let winners run)
        holding_bonus = self._calculate_holding_bonus()
        total_reward += holding_bonus * self.reward_weights['holding_bonus']
        
        # 4. ACTION PENALTY (Trading fees)
        action_penalty = self._calculate_action_penalty(action, price)
        total_reward += action_penalty * self.reward_weights['action_penalty']
        
        return total_reward
    
    def _calculate_pnl_reward(self, action: int, price: float) -> float:
        """Calculate PnL-based reward component."""
        action_type = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}[action]
        position = self.exchange_manager.current_position
        
        if action_type == 'HOLD':
            # Unrealized PnL for holding
            if not np.isclose(position, 0.0) and not np.isclose(self.entry_price, 0.0):
                if position > 0:
                    pnl = (price - self.entry_price) * position
                else:
                    pnl = (self.entry_price - price) * abs(position)
                
                reward = np.tanh(pnl / self.initial_balance) * 10.0  # Scale up
                return reward
            return 0.0
        
        # Execute trade
        position_value = position * price
        size_usd = self.risk_manager.calculate_position_size_usd(
            price,
            position_value,
            self.config.kelly_criterion_fraction
        )
        
        if size_usd < 10.0:
            return 0.0
        
        success, result = self.exchange_manager.execute_order(action_type, price, size_usd)
        
        if not success:
            return -0.01
        
        new_position = self.exchange_manager.current_position
        reward = 0.0
        
        # Position closed (REALIZED PnL)
        if np.isclose(new_position, 0.0) and not np.isclose(position, 0.0):
            if position > 0:
                pnl = (price - self.entry_price) * abs(position)
            else:
                pnl = (self.entry_price - price) * abs(position)
            
            commission = abs(position) * price * self.commission
            net_pnl = pnl - commission
            
            # Update net worth
            self.risk_manager.net_worth += net_pnl
            self.risk_manager.update_metrics(self.risk_manager.net_worth)
            
            # BIG reward for realized PnL (scaled up significantly)
            pnl_pct = net_pnl / self.initial_balance
            reward = pnl_pct * 10000.0  # Very high weight on profit
            
            # Track win/loss streak
            if net_pnl > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self.winning_trades += 1
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
            
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
            
            commission = abs(new_position) * price * self.commission
            self.risk_manager.net_worth -= commission
            
            # Small penalty for opening
            reward = -self.commission * 5.0
            
            self.trade_log.append({
                'step': self.current_step,
                'action': action_type,
                'price': price,
                'size': new_position,
                'net_worth': self.risk_manager.net_worth
            })
        
        return reward
    
    def _calculate_opportunity_cost(self, action: int) -> float:
        """
        Penalty for HOLDING when there's a strong trend.
        
        THIS IS KEY: Prevents agent from learning to always hold!
        """
        # Only penalize HOLD action
        if action != 0:
            return 0.0
        
        # Calculate trend strength
        trend = self._calculate_trend()
        
        # No penalty if already positioned with trend
        position = self.exchange_manager.current_position
        if position > 0 and trend > 0:
            return 0.0  # Long in uptrend = good
        if position < 0 and trend < 0:
            return 0.0  # Short in downtrend = good
        
        # Penalty for being flat or wrong-sided during trend
        if abs(trend) > 0.015:  # 1.5% trend threshold
            penalty = -abs(trend) * 50.0
            
            # Extra penalty if positioned against trend
            if (position > 0 and trend < 0) or (position < 0 and trend > 0):
                penalty *= 2.0
            
            return penalty
        
        return 0.0
    
    def _calculate_holding_bonus(self) -> float:
        """
        Reward HOLDING winning positions (let winners run).
        """
        position = self.exchange_manager.current_position
        
        if np.isclose(position, 0.0) or np.isclose(self.entry_price, 0.0):
            return 0.0
        
        current_price = self.data_source.get_market_price(self.current_step)
        
        # Calculate unrealized profit percentage
        if position > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        
        # Only reward if profitable above threshold
        if pnl_pct > 0.02:  # 2% profit threshold
            # Reward scales with profit and time held
            holding_reward = pnl_pct * 100.0
            
            # Time bonus (capped at 50%)
            time_bonus = min(self.time_in_position / 100.0, 0.5)
            holding_reward *= (1.0 + time_bonus)
            
            return holding_reward
        
        return 0.0
    
    def _calculate_action_penalty(self, action: int, price: float) -> float:
        """Penalize trading to discourage overtrading."""
        if action == 0:  # HOLD
            return 0.0
        
        position = self.exchange_manager.current_position
        position_value = abs(position) * price
        
        # Penalty proportional to position size
        if position_value > 0:
            penalty = -(position_value * self.commission) / self.initial_balance * 100
            return penalty
        
        return 0.0
    
    def _calculate_trend(self) -> float:
        """
        Calculate trend strength from recent price data.
        
        Returns: -1 to 1 (negative=downtrend, positive=uptrend)
        """
        lookback = min(50, self.current_step - self.config.window_size)
        if lookback < 10:
            return 0.0
        
        start_idx = self.current_step - lookback
        end_idx = self.current_step
        
        prices = [self.data_source.get_market_price(i) for i in range(start_idx, end_idx)]
        
        # Linear regression slope
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # Normalize by price
        trend = slope / (np.mean(prices) + 1e-9)
        
        # Clip to [-1, 1]
        return float(np.clip(trend * 100, -1.0, 1.0))
    
    def _calculate_volatility(self) -> float:
        """Calculate recent volatility."""
        if len(self.recent_returns) < 2:
            return 0.02  # Default
        
        return float(np.std(list(self.recent_returns)))
    
    def _get_obs(self, index: int) -> Dict[str, np.ndarray]:
        """Get observation at given step."""
        features = self.data_source.get_latest_observation()
        
        norm_net_worth = self.risk_manager.net_worth / self.initial_balance
        
        max_position_value = self.initial_balance * self.config.max_position_leverage
        current_price = self.data_source.get_market_price(min(index, len(self.data_source.data_store) - 1))
        position_value = self.exchange_manager.current_position * current_price
        norm_position = position_value / max_position_value
        
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
    
    def _check_stop_loss_take_profit(self, price: float) -> float:
        """Check and execute stop-loss or take-profit."""
        position = self.exchange_manager.current_position
        
        if np.isclose(position, 0.0) or np.isclose(self.entry_price, 0.0):
            return 0.0
        
        reward = 0.0
        
        if position > 0:
            pnl_pct = (price - self.entry_price) / self.entry_price
            
            if pnl_pct <= -self.config.stop_loss_pct:
                logger.info(f"🛑 Stop-loss triggered: {pnl_pct:.2%}")
                success, _ = self.exchange_manager.execute_order('SELL', price, position * price)
                if success:
                    reward = -10.0
            
            elif pnl_pct >= self.config.take_profit_pct:
                logger.info(f"💰 Take-profit triggered: {pnl_pct:.2%}")
                success, _ = self.exchange_manager.execute_order('SELL', price, position * price)
                if success:
                    reward = 5.0
        
        else:
            pnl_pct = (self.entry_price - price) / self.entry_price
            
            if pnl_pct <= -self.config.stop_loss_pct:
                logger.info(f"🛑 Stop-loss triggered: {pnl_pct:.2%}")
                success, _ = self.exchange_manager.execute_order('BUY', price, abs(position) * price)
                if success:
                    reward = -10.0
            
            elif pnl_pct >= self.config.take_profit_pct:
                logger.info(f"💰 Take-profit triggered: {pnl_pct:.2%}")
                success, _ = self.exchange_manager.execute_order('BUY', price, abs(position) * price)
                if success:
                    reward = 5.0
        
        return reward
    
    def _close_final_position(self, price: float) -> float:
        """Close all positions at episode end."""
        position = self.exchange_manager.current_position
        
        if np.isclose(position, 0.0):
            final_return = (self.risk_manager.net_worth - self.initial_balance) / self.initial_balance
            return final_return * 100.0
        
        if position > 0:
            pnl = (price - self.entry_price) * position
        else:
            pnl = (self.entry_price - price) * abs(position)
        
        commission = abs(position) * price * self.commission
        net_pnl = pnl - commission
        
        self.risk_manager.net_worth += net_pnl
        self.exchange_manager.current_position = 0.0
        
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
            "equity_curve": self.equity_curve[-100:],
            "volatility_factor": self.curriculum.get_volatility_factor(),
            # NEW: Additional info
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "winning_trades": self.winning_trades,
            "trend": self._calculate_trend(),
            "volatility": self._calculate_volatility()
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
            print(f"Trades: {info['trade_count']} (Win streak: {info['consecutive_wins']})")
            print(f"Episode Reward: {info['episode_reward']:.2f}")
            print(f"Trend: {info['trend']:+.3f} | Volatility: {info['volatility']:.3f}")
            print(f"{'='*60}\n")
    
    def get_episode_statistics(self) -> Dict:
        """Calculate episode performance statistics."""
        equity = np.array(self.equity_curve)
        
        returns = np.diff(equity) / equity[:-1]
        
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino = 0.0
        
        winning_trades = sum(1 for t in self.trade_log if t.get('pnl', 0) > 0)
        total_trades = len([t for t in self.trade_log if 'pnl' in t])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': self.risk_manager.max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_net_worth': equity[-1],
            'winning_trades': self.winning_trades
        }
