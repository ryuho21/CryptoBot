# advanced_main.py - COMPLETE ADVANCED TRAINING SYSTEM
"""
Enhanced training system integrating:
- Multi-asset support
- Market regime detection
- Order book analysis
- A/B testing
- Advanced portfolio management
"""

import sys
import os
import argparse
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional
import logging
import time
import tempfile
from model_core import PolicyIQNNet, WorldModel, RNDNet, init_hidden_state
from ppo_buffer import PPOBuffer

logger = logging.getLogger(__name__)

from config import get_config, Config
from data_pipeline import ExchangeDataSource
from exchange_manager import RiskManager, ExchangeManager
from trading_env import TradingBotEnv
from ppo_agent import PPOAgent
from notification_service import NotificationService
from logger import LOGGER as log

# Advanced features
from multi_asset_manager import MultiAssetAgent, PortfolioOptimizer, RebalancingScheduler
from market_regime_detector import RegimeDetector, RegimeAdaptiveStrategy
from order_book_analyzer import OrderBookAnalyzer
from backtesting_framework import BacktestEngine
from ab_testing_framework import ABTestFramework

# Import original main components
from main import (initialize_system, train_loop, evaluate_agent, 
                 JSONLLogger, MetricsTracker)


class AdvancedTradingSystem:
    """
    Complete advanced trading system with all features.
    
    Integrates:
    - Multi-asset portfolio management
    - Market regime adaptation
    - Order book analysis
    - Comprehensive testing
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Core components
        self.env = None
        self.agent = None
        self.logger = None
        self.notifier = None
        self.metrics = None
        
        # Advanced components
        self.multi_asset_agent = None
        self.regime_strategy = None
        self.orderbook_analyzer = None
        self.backtest_engine = None
        self.ab_framework = None
        
        self.initialized = False
    
    def initialize(self, symbols: Optional[List[str]] = None):
        """
        Initialize all system components.
        
        Args:
            symbols: list of symbols for multi-asset (None = single asset)
        """
        log.info("🚀 Initializing Advanced Trading System...")
        
        # Core initialization
        self.env, self.agent, self.logger, self.notifier, self.metrics = initialize_system(self.config)
        
        # Multi-asset setup
        if symbols and len(symbols) > 1:
            log.info(f"📊 Initializing multi-asset system: {symbols}")
            self.multi_asset_agent = MultiAssetAgent(self.config, symbols)
        
        # Market regime detection
        log.info("🔍 Initializing market regime detector...")
        self.regime_strategy = RegimeAdaptiveStrategy(self.config)
        
        # Order book analyzer
        log.info("📖 Initializing order book analyzer...")
        self.orderbook_analyzer = OrderBookAnalyzer(self.config)
        
        # Testing frameworks
        log.info("🧪 Initializing testing frameworks...")
        self.backtest_engine = BacktestEngine(self.config)
        self.ab_framework = ABTestFramework(self.config)
        
        self.initialized = True
        log.info("✅ Advanced Trading System initialized")
    
    def train_with_regime_adaptation(self):
        """Enhanced training with market regime adaptation and periodic checkpointing."""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        log.info("🎯 Starting regime-adaptive training...")
        
        # Get price history for regime detection
        price_history = []
        
        total_timesteps = 0
        episode = 0
        
        # Checkpoint configuration
        checkpoint_frequency = 100  # Save every 100 episodes
        last_checkpoint_episode = 0
        
        while total_timesteps < self.config.training_timesteps:
            obs, info = self.env.reset()
            self.agent.reset_recurrent_state()
            done = False
            episode_reward = 0.0
            step_count = 0
            
            # Update market regime
            if len(price_history) > 100:
                prices = np.array(price_history[-100:])
                self.regime_strategy.update_regime(prices)
                
                # Get regime-adjusted parameters
                regime_params = self.regime_strategy.get_adjusted_parameters()
                
                # Apply to environment
                self.env.config.stop_loss_pct = regime_params['stop_loss_pct']
                self.env.config.take_profit_pct = regime_params['take_profit_pct']
                
                log.info(f"📊 Regime: {self.regime_strategy.current_regime.value}, "
                        f"SL={regime_params['stop_loss_pct']:.2%}, "
                        f"TP={regime_params['take_profit_pct']:.2%}")
            
            while not done and step_count < self.env.max_steps:
                # Standard training step
                h_in = tuple(t.clone() for t in self.agent.h_state)
                action, log_prob, value = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Track price
                price_history.append(info['current_price'])
                
                # Compute intrinsic reward
                next_obs_feat = next_obs["features"][-1, :]
                intrinsic_reward = self.agent.compute_intrinsic_reward(next_obs_feat)
                total_reward = reward + intrinsic_reward
                
                # Store transition
                self.agent.buffer.add(
                    obs, next_obs_feat, action, log_prob,
                    total_reward, done, h_in, intrinsic_reward, value
                )
                
                obs = next_obs
                episode_reward += reward
                step_count += 1
                total_timesteps += 1
                
                # PPO update
                if self.agent.buffer.is_full():
                    losses = self.agent.learn(obs, done)
                    self.metrics.record_losses(*losses)
                    self.agent.buffer.reset()
            
            # Episode complete
            episode_stats = self.env.get_episode_statistics()
            episode_stats['episode_reward'] = episode_reward
            episode_stats['episode_length'] = step_count
            
            self.metrics.record_episode(episode_stats)
            self.env.curriculum.update(episode_stats)
            
            log.info(f"✅ Episode {episode + 1}: Reward={episode_reward:.2f}, "
                    f"Return={episode_stats['total_return']:.2%}, "
                    f"Regime={self.regime_strategy.current_regime.value if hasattr(self.regime_strategy, 'current_regime') else 'N/A'}")
            
            episode += 1
            
            # PERIODIC CHECKPOINTING
            if episode - last_checkpoint_episode >= checkpoint_frequency:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir, 
                    f"checkpoint_ep{episode}.pt"
                )
                try:
                    saved_path = self.agent.save_checkpoint(checkpoint_path)
                    if saved_path:
                        log.info(f"💾 Checkpoint saved at episode {episode}: {saved_path}")
                        last_checkpoint_episode = episode
                except Exception as e:
                    log.warning(f"⚠️ Could not save checkpoint: {e}")
        
        log.info("✅ Regime-adaptive training complete")
        
        # FINAL MODEL SAVE with multiple fallback locations
        log.info("💾 Saving final model with fallback locations...")
        import time
        
        final_save_locations = [
            os.path.join(self.config.checkpoint_dir, "final_model.pt"),
            "/tmp/final_model_last.pt",
            f"/tmp/final_model_{int(time.time())}.pt",
            "/tmp/final_model_latest.pt"
        ]
        
        saved = False
        for save_path in final_save_locations:
            try:
                result = self.agent.save_model(save_path)
                if result:
                    log.info(f"✅ Final model saved to: {result}")
                    saved = True
                    break
            except Exception as e:
                log.debug(f"Could not save to {save_path}: {e}")
        
        if not saved:
            log.error("❌ Failed to save final model to any location!")
            log.info(f"💡 Last checkpoint is available at episode {last_checkpoint_episode}")
            log.info(f"   Path: {self.config.checkpoint_dir}/checkpoint_ep{last_checkpoint_episode}.pt")
        
        return self.metrics
        
    def run_comprehensive_backtest(self, agent: Optional[PPOAgent] = None):
        """Run comprehensive backtesting suite."""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        test_agent = agent or self.agent
        
        log.info("🔄 Running comprehensive backtest suite...")
        
        # Single backtest
        result = self.backtest_engine.run_backtest(test_agent)
        self.backtest_engine.generate_report(result, save_dir="runs/backtest/single")
        
        # Walk-forward analysis
        log.info("📊 Running walk-forward analysis...")
        wf_results = self.backtest_engine.walk_forward_analysis(
            test_agent,
            train_periods=1000,
            test_periods=500,
            step_periods=250
        )
        
        # Monte Carlo simulation
        log.info("🎲 Running Monte Carlo simulation...")
        if len(result.returns) > 0:
            mc_results = self.backtest_engine.monte_carlo_simulation(
                np.array(result.returns),
                initial_capital=self.config.initial_balance,
                num_simulations=1000
            )
            
            log.info(f"Monte Carlo Results:")
            log.info(f"  Mean final value: ${mc_results['mean_final_value']:,.2f}")
            log.info(f"  Probability of profit: {mc_results['prob_profit']:.2%}")
            log.info(f"  5th percentile: ${mc_results['percentile_5']:,.2f}")
            log.info(f"  95th percentile: ${mc_results['percentile_95']:,.2f}")
        
        log.info("✅ Comprehensive backtest complete")
        
        return result, wf_results
    
    def run_ab_test(self, agents: Dict[str, PPOAgent], num_trials: int = 10):
        """Run A/B test comparing multiple agents."""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        log.info(f"🧪 Running A/B test with {len(agents)} variants...")
        
        # Run test
        results = self.ab_framework.run_ab_test(agents, num_trials=num_trials)
        
        # Statistical comparisons
        variant_names = list(results.keys())
        if len(variant_names) >= 2:
            for i in range(len(variant_names) - 1):
                comparison = self.ab_framework.statistical_comparison(
                    variant_names[i],
                    variant_names[i + 1]
                )
        
        # Save results
        self.ab_framework.save_results(save_dir="runs/ab_test")
        
        log.info("✅ A/B test complete")
        
        return results


def main():
    """Main entry point for advanced trading system."""
    parser = argparse.ArgumentParser(description="Advanced Trading Bot System")
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--preset', type=str, choices=['development', 'production', 'testing'],
                       help='Configuration preset')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'backtest', 'ab_test', 'live'],
                       help='Execution mode')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--multi-asset', nargs='+', help='Symbols for multi-asset trading')
    parser.add_argument('--ab-variants', nargs='+', help='Checkpoint paths for A/B testing')
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(preset=args.preset, config_path=args.config)
    config.print_summary()
    
    # Validate
    valid, errors = config.validate()
    if not valid:
        log.error("❌ Configuration validation failed:")
        for error in errors:
            log.error(f"  • {error}")
        sys.exit(1)
    
    # Initialize system
    system = AdvancedTradingSystem(config)
    system.initialize(symbols=args.multi_asset)
    
    # Resume from checkpoint
    if args.resume:
        log.info(f"📂 Resuming from checkpoint: {args.resume}")
        system.agent.load_checkpoint(args.resume)
    
    # Execute based on mode
    if args.mode == 'train':
        log.info("🎯 Mode: Training with regime adaptation")
        metrics = system.train_with_regime_adaptation()
        
        # Save final model
        model_path = os.path.join(config.checkpoint_dir, "final_model.pt")
        system.agent.save_model(model_path)
        log.info(f"💾 Model saved: {model_path}")
    
    elif args.mode == 'backtest':
        log.info("🔍 Mode: Comprehensive backtesting")
        result, wf_results = system.run_comprehensive_backtest()
        
        log.info(f"\n{'='*70}")
        log.info("BACKTEST SUMMARY")
        log.info(f"{'='*70}")
        log.info(f"Total Return: {result.total_return:.2%}")
        log.info(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        log.info(f"Max Drawdown: {result.max_drawdown:.2%}")
        log.info(f"Win Rate: {result.win_rate:.2%}")
        log.info(f"{'='*70}\n")
    
    elif args.mode == 'ab_test':
        log.info("🧪 Mode: A/B Testing")
        
        if not args.ab_variants:
            log.error("❌ A/B testing requires --ab-variants")
            sys.exit(1)
        
        # Load agents from checkpoints
        agents = {}
        for i, checkpoint_path in enumerate(args.ab_variants):
            agent = PPOAgent(config, system.env)
            agent.load_checkpoint(checkpoint_path)
            agents[f"variant_{i}"] = agent
        
        results = system.run_ab_test(agents, num_trials=10)
        
        # Print winner
        best_variant = max(results.items(), key=lambda x: x[1].mean_sharpe)
        
        log.info(f"\n{'='*70}")
        log.info("A/B TEST RESULTS")
        log.info(f"{'='*70}")
        log.info(f"🏆 Winner: {best_variant[0]}")
        log.info(f"  Mean Return: {best_variant[1].mean_return:.2%}")
        log.info(f"  Mean Sharpe: {best_variant[1].mean_sharpe:.2f}")
        log.info(f"  Win Rate: {best_variant[1].mean_win_rate:.2%}")
        log.info(f"{'='*70}\n")
    
    elif args.mode == 'live':
        log.info("🔴 Mode: Live Trading")
        log.warning("⚠️ Live trading mode - USE WITH CAUTION!")
        
        # Initialize live bridge
        from live_bridge import LiveTradingBridge
        
        bridge = LiveTradingBridge(config, system.env.risk_manager, system.notifier)
        bridge.start()
        
        try:
            # Live trading loop
            log.info("🚀 Starting live trading...")
            system.notifier.send_critical("🔴 Live trading started!")
            
            obs, _ = system.env.reset()
            system.agent.reset_recurrent_state()
            
            step = 0
            max_steps = 10000  # Safety limit
            
            while step < max_steps:
                # Get latest market data
                latest_data = bridge.get_latest_ohlcv(config.symbol, limit=config.window_size)
                
                if latest_data is None:
                    log.warning("⚠️ No market data available, waiting...")
                    time.sleep(5)
                    continue
                
                # Update order book if available
                # (Simulated - real implementation would fetch from exchange)
                
                # Select action
                action, log_prob, value = system.agent.select_action(obs, deterministic=True)
                
                # Execute action
                action_names = ['HOLD', 'BUY', 'SELL']
                action_name = action_names[action]
                
                if action_name != 'HOLD':
                    log.info(f"📊 Action: {action_name} (confidence: {np.exp(log_prob):.2%})")
                    
                    # Execute order through bridge
                    if action_name == 'BUY':
                        result = bridge.create_market_order(config.symbol, 'buy', 0.001)
                    elif action_name == 'SELL':
                        result = bridge.create_market_order(config.symbol, 'sell', 0.001)
                    
                    log.info(f"Order result: {result}")
                
                # Update observation (simplified)
                obs = system.env._get_obs(system.env.current_step)
                
                step += 1
                time.sleep(60)  # Wait 1 minute between decisions
            
        except KeyboardInterrupt:
            log.info("🛑 Live trading interrupted by user")
            system.notifier.send_warning("⚠️ Live trading stopped by user")
        
        finally:
            bridge.stop()
            log.info("👋 Live trading stopped")
    
    log.info("✅ Advanced Trading System execution complete")

class MarketScenarioGenerator:
    """
    Generate diverse training scenarios from base market data.
    
    Usage:
        generator = MarketScenarioGenerator(historical_ohlcv_df)
        scenarios = generator.generate_all_scenarios()
        
        for name, data in scenarios:
            env.load_data(data)
            # Train for N episodes
    """
    
    def __init__(self, base_df: pd.DataFrame):
        """
        Initialize with base OHLCV data.
        
        Args:
            base_df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.base = base_df.copy()
        
        # Validate columns
        required_cols = ['open', 'high', 'low', 'close']
        missing = [col for col in required_cols if col not in self.base.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        logger.info(f"📊 MarketScenarioGenerator initialized with {len(self.base)} candles")
    
    def generate_all_scenarios(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        Generate all market scenarios.
        
        Returns:
            List of (scenario_name, dataframe) tuples
        """
        scenarios = []
        
        # 1. Original data
        scenarios.append(('original', self.base.copy()))
        logger.info("  ✓ Original scenario")
        
        # 2. Bull run
        bull = self._create_bull_run()
        scenarios.append(('bull_run', bull))
        logger.info("  ✓ Bull run scenario (+30%)")
        
        # 3. Bear market
        bear = self._create_bear_market()
        scenarios.append(('bear_market', bear))
        logger.info("  ✓ Bear market scenario (-25%)")
        
        # 4. High volatility
        volatile = self._create_high_volatility()
        scenarios.append(('high_volatility', volatile))
        logger.info("  ✓ High volatility scenario (±5%)")
        
        # 5. Range bound
        ranging = self._create_range_bound()
        scenarios.append(('range_bound', ranging))
        logger.info("  ✓ Range-bound scenario")
        
        # 6. Flash crash
        crash = self._create_flash_crash()
        scenarios.append(('flash_crash', crash))
        logger.info("  ✓ Flash crash scenario (-15% → recovery)")
        
        # 7. Trend reversal
        reversal = self._create_trend_reversal()
        scenarios.append(('trend_reversal', reversal))
        logger.info("  ✓ Trend reversal scenario")
        
        # 8. Whipsaw
        whipsaw = self._create_whipsaw()
        scenarios.append(('whipsaw', whipsaw))
        logger.info("  ✓ Whipsaw scenario (false breakouts)")
        
        logger.info(f"📊 Generated {len(scenarios)} market scenarios")
        
        return scenarios
    
    def _create_bull_run(self, gain_pct: float = 0.30) -> pd.DataFrame:
        """Create steady bull run scenario."""
        df = self.base.copy()
        multiplier = np.linspace(1.0, 1.0 + gain_pct, len(df))
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * multiplier
        
        return df
    
    def _create_bear_market(self, loss_pct: float = 0.25) -> pd.DataFrame:
        """Create steady bear market scenario."""
        df = self.base.copy()
        multiplier = np.linspace(1.0, 1.0 - loss_pct, len(df))
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * multiplier
        
        return df
    
    def _create_high_volatility(self, noise_std: float = 0.025) -> pd.DataFrame:
        """Create high volatility with random noise."""
        df = self.base.copy()
        
        # Set seed for reproducibility
        np.random.seed(42)
        noise = np.random.normal(0, noise_std, len(df))
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * (1 + noise)
        
        # Ensure OHLC relationship is maintained
        df = self._fix_ohlc_relationship(df)
        
        return df
    
    def _create_range_bound(self, range_pct: float = 0.03) -> pd.DataFrame:
        """Create oscillating range-bound market."""
        df = self.base.copy()
        
        mean_price = df['close'].mean()
        
        # Create sine wave oscillation
        periods = 6  # Number of complete cycles
        sine_wave = range_pct * mean_price * np.sin(
            np.linspace(0, periods * 2 * np.pi, len(df))
        )
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = mean_price + sine_wave
        
        return df
    
    def _create_flash_crash(self, crash_pct: float = 0.15) -> pd.DataFrame:
        """Create sudden crash with recovery."""
        df = self.base.copy()
        
        # Crash happens at 1/3 point
        crash_point = len(df) // 3
        # Recovery happens at 1/4 later
        recovery_point = crash_point + len(df) // 4
        
        multiplier = np.ones(len(df))
        
        # Normal before crash
        multiplier[:crash_point] = 1.0
        
        # Sudden crash
        multiplier[crash_point:recovery_point] = np.linspace(
            1.0, 1.0 - crash_pct, recovery_point - crash_point
        )
        
        # Gradual recovery (but not full)
        multiplier[recovery_point:] = np.linspace(
            1.0 - crash_pct, 1.0 - (crash_pct * 0.3), len(df) - recovery_point
        )
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * multiplier
        
        return df
    
    def _create_trend_reversal(self) -> pd.DataFrame:
        """Create uptrend → downtrend reversal."""
        df = self.base.copy()
        
        mid = len(df) // 2
        
        # First half: uptrend
        multiplier_up = np.linspace(1.0, 1.15, mid)
        # Second half: downtrend
        multiplier_down = np.linspace(1.0, 0.90, len(df) - mid)
        
        multiplier = np.concatenate([multiplier_up, multiplier_down])
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * multiplier
        
        return df
    
    def _create_whipsaw(self, num_whipsaws: int = 5) -> pd.DataFrame:
        """Create false breakout pattern (whipsaw)."""
        df = self.base.copy()
        
        mean_price = df['close'].mean()
        
        # Create sharp oscillations
        segment_length = len(df) // num_whipsaws
        multiplier = np.ones(len(df))
        
        for i in range(num_whipsaws):
            start = i * segment_length
            end = min((i + 1) * segment_length, len(df))
            
            # Alternate between spikes up and down
            if i % 2 == 0:
                # Spike up then revert
                spike = np.concatenate([
                    np.linspace(1.0, 1.05, (end - start) // 2),
                    np.linspace(1.05, 1.0, (end - start) // 2)
                ])
            else:
                # Spike down then revert
                spike = np.concatenate([
                    np.linspace(1.0, 0.95, (end - start) // 2),
                    np.linspace(0.95, 1.0, (end - start) // 2)
                ])
            
            multiplier[start:end] = spike[:end - start]
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * multiplier
        
        return df
    
    def _fix_ohlc_relationship(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLC maintains correct relationships (H >= O,C,L and L <= O,C,H)."""
        # High should be max of O, H, C
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        
        # Low should be min of O, L, C
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def create_custom_scenario(
        self,
        name: str,
        trend: str = 'neutral',
        volatility: str = 'normal',
        events: List[dict] = None
    ) -> Tuple[str, pd.DataFrame]:
        """
        Create custom scenario.
        
        Args:
            name: Scenario name
            trend: 'up', 'down', 'neutral'
            volatility: 'low', 'normal', 'high'
            events: List of {'type': 'crash'/'pump', 'start': 0.5, 'magnitude': 0.1}
        
        Returns:
            (name, dataframe) tuple
        """
        df = self.base.copy()
        
        # Apply trend
        if trend == 'up':
            multiplier = np.linspace(1.0, 1.20, len(df))
        elif trend == 'down':
            multiplier = np.linspace(1.0, 0.85, len(df))
        else:
            multiplier = np.ones(len(df))
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * multiplier
        
        # Apply volatility
        if volatility == 'high':
            noise = np.random.normal(0, 0.03, len(df))
        elif volatility == 'low':
            noise = np.random.normal(0, 0.005, len(df))
        else:
            noise = np.random.normal(0, 0.015, len(df))
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * (1 + noise)
        
        # Apply events
        if events:
            for event in events:
                event_type = event.get('type', 'crash')
                start_pct = event.get('start', 0.5)
                magnitude = event.get('magnitude', 0.1)
                
                start_idx = int(len(df) * start_pct)
                duration = min(100, len(df) - start_idx)
                
                event_mult = np.ones(len(df))
                if event_type == 'crash':
                    event_mult[start_idx:start_idx + duration] = np.linspace(
                        1.0, 1.0 - magnitude, duration
                    )
                elif event_type == 'pump':
                    event_mult[start_idx:start_idx + duration] = np.linspace(
                        1.0, 1.0 + magnitude, duration
                    )
                
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = df[col] * event_mult
        
        df = self._fix_ohlc_relationship(df)
        
        return (name, df)
    
    def get_scenario_statistics(self, scenarios: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Calculate statistics for all scenarios.
        
        Returns:
            DataFrame with scenario statistics
        """
        stats = []
        
        for name, df in scenarios:
            returns = df['close'].pct_change().dropna()
            
            stat = {
                'scenario': name,
                'length': len(df),
                'total_return': (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100,
                'volatility': returns.std() * np.sqrt(len(df)) * 100,
                'max_drawdown': self._calculate_max_drawdown(df['close']) * 100,
                'mean_price': df['close'].mean(),
                'min_price': df['close'].min(),
                'max_price': df['close'].max(),
            }
            stats.append(stat)
        
        return pd.DataFrame(stats)
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown.min()


# ========== USAGE EXAMPLE ==========
if __name__ == '__main__':
    # Example usage
    print("📊 Market Scenario Generator")
    print("=" * 70)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    base_price = 40000
    noise = np.random.normal(0, 100, 1000)
    
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': base_price + noise,
        'high': base_price + noise + 50,
        'low': base_price + noise - 50,
        'close': base_price + noise + np.random.normal(0, 20, 1000),
        'volume': np.random.uniform(1, 100, 1000)
    })
    
    # Generate scenarios
    generator = MarketScenarioGenerator(sample_df)
    scenarios = generator.generate_all_scenarios()
    
    # Print statistics
    stats = generator.get_scenario_statistics(scenarios)
    print("\n📈 Scenario Statistics:")
    print(stats.to_string(index=False))
    
    print("\n✅ All scenarios generated successfully!")
    print(f"   Use these to train your agent on diverse market conditions")
if __name__ == "__main__":
    main()
    
