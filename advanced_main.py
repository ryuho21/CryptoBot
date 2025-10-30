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


if __name__ == "__main__":
    main()