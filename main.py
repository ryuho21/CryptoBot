# main.py - PRODUCTION-READY TRAINING LOOP
"""
Complete training system with:
- Proper initialization
- Checkpointing
- Evaluation
- Metrics tracking
- Error handling
- Graceful shutdown
"""

import pandas as pd
import numpy as np
import time
import signal
import sys
from datetime import datetime
import json
import os
from typing import Optional
import logging

from config import get_config, Config
from data_pipeline import ExchangeDataSource
from exchange_manager import RiskManager, ExchangeManager
from trading_env import TradingBotEnv
from ppo_agent import PPOAgent
from notification_service import NotificationService
from logger import LOGGER as log

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    global SHUTDOWN_REQUESTED
    print("\n🛑 Shutdown requested... Saving checkpoint...")
    SHUTDOWN_REQUESTED = True


signal.signal(signal.SIGINT, signal_handler)


class JSONLLogger:
    """JSONL logger for structured logging with permission fallback."""

    def __init__(self, path: str):
        self.path = path
        log_dir = os.path.dirname(path)

        # Try to create log directory
        try:
            os.makedirs(log_dir, exist_ok=True)
        except PermissionError:
            alt_path = os.path.join("/tmp", os.path.basename(path))
            print(f"⚠️ Permission denied creating {log_dir}, using {alt_path} instead.")
            self.path = alt_path
            log_dir = "/tmp"

        # Try to create or clear file
        try:
            with open(self.path, "w") as f:
                pass
        except PermissionError:
            alt_path = os.path.join("/tmp", os.path.basename(self.path))
            print(f"⚠️ Permission denied writing {self.path}, using {alt_path} instead.")
            self.path = alt_path
            with open(self.path, "w") as f:
                pass

        log.info(f"📝 JSONL logger initialized: {self.path}")

    def log_step(self, data: dict):
        """Log single step."""
        data["timestamp"] = datetime.now().isoformat()
        with open(self.path, "a") as f:
            f.write(json.dumps(data) + "\n")

    def log_episode(self, data: dict):
        """Log episode summary."""
        self.log_step({"type": "episode", **data})



class MetricsTracker:
    """Track and compute performance metrics."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_sharpes = []
        self.episode_win_rates = []
        self.losses = {'total': [], 'policy': [], 'critic': [], 'wm': [], 'rnd': []}
    
    def record_episode(self, stats: dict):
        """Record episode statistics."""
        self.episode_rewards.append(stats.get('episode_reward', 0.0))
        self.episode_returns.append(stats.get('total_return', 0.0))
        self.episode_lengths.append(stats.get('episode_length', 0))
        self.episode_sharpes.append(stats.get('sharpe_ratio', 0.0))
        self.episode_win_rates.append(stats.get('win_rate', 0.0))
    
    def record_losses(self, total, policy, critic, wm, rnd):
        """Record training losses."""
        self.losses['total'].append(total)
        self.losses['policy'].append(policy)
        self.losses['critic'].append(critic)
        self.losses['wm'].append(wm)
        self.losses['rnd'].append(rnd)
    
    def get_recent_stats(self, window: int = 10) -> dict:
        """Get statistics over recent episodes."""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-window:]
        recent_returns = self.episode_returns[-window:]
        recent_sharpes = self.episode_sharpes[-window:]
        recent_win_rates = self.episode_win_rates[-window:]
        
        return {
            'mean_reward': np.mean(recent_rewards),
            'mean_return': np.mean(recent_returns),
            'mean_sharpe': np.mean(recent_sharpes),
            'mean_win_rate': np.mean(recent_win_rates),
            'std_reward': np.std(recent_rewards),
            'total_episodes': len(self.episode_rewards)
        }
    
    def save(self, path: str):
        """Save metrics to file."""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_returns': self.episode_returns,
            'episode_sharpes': self.episode_sharpes,
            'episode_win_rates': self.episode_win_rates,
            'losses': self.losses
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        log.info(f"💾 Metrics saved: {path}")


def initialize_system(config: Config):
    """
    Initialize all system components.
    
    Returns:
        Tuple of (env, agent, logger, notifier, metrics)
    """
    log.info("🚀 Initializing trading system...")
    
    # Create directories
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.runs_dir, exist_ok=True)
    
    # Initialize components
    risk_manager = RiskManager(config)
    exchange_manager = ExchangeManager(risk_manager, config)
    data_source = ExchangeDataSource(config)
    
    # Load data
    log.info("📊 Loading market data...")
    data_source.load_initial_data()
    
    # Create environment and agent
    log.info("🎮 Creating environment...")
    env = TradingBotEnv(data_source, exchange_manager, config)
    
    log.info("🤖 Initializing agent...")
    agent = PPOAgent(config, env)
    
    # Utilities
    logger = JSONLLogger(os.path.join(config.runs_dir, "training_log.jsonl"))
    notifier = NotificationService(config)
    metrics = MetricsTracker()
    
    # Send startup notification
    notifier.send_update(
        f"🚀 Training started\n"
        f"Symbol: {config.symbol}\n"
        f"Mode: {'🧪 Testnet' if config.use_testnet else '🔴 Live'}\n"
        f"Device: {config.device}\n"
        f"Total steps: {config.training_timesteps:,}"
    )
    
    log.info("✅ System initialization complete")
    
    return env, agent, logger, notifier, metrics


def evaluate_agent(agent: PPOAgent, env: TradingBotEnv, num_episodes: int = 5) -> dict:
    """
    Evaluate agent performance.
    
    Args:
        agent: trained agent
        env: trading environment
        num_episodes: number of evaluation episodes
    
    Returns:
        dict with evaluation metrics
    """
    log.info(f"🔍 Evaluating agent over {num_episodes} episodes...")
    
    eval_rewards = []
    eval_returns = []
    eval_sharpes = []
    eval_win_rates = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        agent.reset_recurrent_state()
        done = False
        episode_reward = 0.0
        
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        # Get episode statistics
        stats = env.get_episode_statistics()
        
        eval_rewards.append(episode_reward)
        eval_returns.append(stats['total_return'])
        eval_sharpes.append(stats['sharpe_ratio'])
        eval_win_rates.append(stats['win_rate'])
    
    eval_stats = {
        'mean_reward': float(np.mean(eval_rewards)),
        'mean_return': float(np.mean(eval_returns)),
        'mean_sharpe': float(np.mean(eval_sharpes)),
        'mean_win_rate': float(np.mean(eval_win_rates)),
        'std_reward': float(np.std(eval_rewards))
    }
    
    log.info(f"📊 Evaluation results: "
            f"Reward={eval_stats['mean_reward']:.2f}, "
            f"Sharpe={eval_stats['mean_sharpe']:.2f}, "
            f"WR={eval_stats['mean_win_rate']:.2%}")
    
    return eval_stats


def train_loop(env: TradingBotEnv, agent: PPOAgent, logger: JSONLLogger,
               notifier: NotificationService, metrics: MetricsTracker, config: Config):
    """
    Main training loop.
    
    Args:
        env: trading environment
        agent: PPO agent
        logger: JSONL logger
        notifier: notification service
        metrics: metrics tracker
        config: configuration
    """
    log.info("🎯 Starting training loop...")
    
    total_timesteps = 0
    episode = 0
    best_sharpe = -np.inf
    best_reward = -np.inf
    
    start_time = time.time()
    last_save_time = start_time
    last_eval_time = start_time
    
    while total_timesteps < config.training_timesteps and not SHUTDOWN_REQUESTED:
        # === EPISODE ROLLOUT ===
        obs, info = env.reset()
        agent.reset_recurrent_state()
        done = False
        episode_reward = 0.0
        step_count = 0
        
        log.info(f"\n{'='*70}")
        log.info(f"📍 Episode {episode + 1} | Total steps: {total_timesteps:,}")
        log.info(f"{'='*70}")
        
        while not done and step_count < env.max_steps:
            # Select action
            h_in = tuple(t.clone() for t in agent.h_state)
            action, log_prob, value = agent.select_action(obs)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Compute intrinsic reward
            next_obs_feat = next_obs["features"][-1, :]
            intrinsic_reward = agent.compute_intrinsic_reward(next_obs_feat)
            total_reward = reward + intrinsic_reward
            
            # Store transition
            agent.buffer.add(
                obs, next_obs_feat, action, log_prob,
                total_reward, done, h_in, intrinsic_reward, value
            )
            
            obs = next_obs
            episode_reward += reward
            step_count += 1
            total_timesteps += 1
            
            # Log step
            logger.log_step({
                "episode": episode,
                "step": total_timesteps,
                "reward": float(reward),
                "intrinsic_reward": float(intrinsic_reward),
                "value": float(value),
                "action": int(action),
                "net_worth": float(info['net_worth']),
                "position": float(info['position'])
            })
            
            # === PPO UPDATE ===
            if agent.buffer.is_full():
                if env.risk_manager.circuit_breaker_tripped:
                    log.warning("⛔ Circuit breaker tripped - skipping update")
                    notifier.send_warning("Circuit breaker tripped")
                    break
                
                # Learn
                total_loss, policy_loss, critic_loss, wm_loss, rnd_loss = agent.learn(obs, done)
                
                # Record losses
                metrics.record_losses(total_loss, policy_loss, critic_loss, wm_loss, rnd_loss)
                
                # Log losses
                log.info(f"📉 Losses: Total={total_loss:.4f}, Policy={policy_loss:.4f}, "
                        f"Critic={critic_loss:.4f}, WM={wm_loss:.4f}, RND={rnd_loss:.4f}")
                
                # Reset buffer
                agent.buffer.reset()
            
            # Check shutdown
            if SHUTDOWN_REQUESTED:
                break
        
        # === EPISODE END ===
        episode_stats = env.get_episode_statistics()
        episode_stats['episode_reward'] = episode_reward
        episode_stats['episode_length'] = step_count
        episode_stats['total_timesteps'] = total_timesteps
        
        # Update curriculum
        env.curriculum.update(episode_stats)
        
        # Record metrics
        metrics.record_episode(episode_stats)
        
        # Log episode
        logger.log_episode(episode_stats)
        
        # Print summary
        log.info(f"\n{'='*70}")
        log.info(f"📊 Episode {episode + 1} Summary:")
        log.info(f"  Reward: {episode_reward:.2f}")
        log.info(f"  Return: {episode_stats['total_return']:.2%}")
        log.info(f"  Sharpe: {episode_stats['sharpe_ratio']:.2f}")
        log.info(f"  Win Rate: {episode_stats['win_rate']:.2%}")
        log.info(f"  Max DD: {episode_stats['max_drawdown']:.2%}")
        log.info(f"  Trades: {episode_stats['total_trades']}")
        log.info(f"  Final NW: ${episode_stats['final_net_worth']:,.2f}")
        log.info(f"{'='*70}\n")
        
        # Notification
        if (episode + 1) % 10 == 0:
            recent_stats = metrics.get_recent_stats(10)
            notifier.send_update(
                f"📈 Episode {episode + 1}\n"
                f"Steps: {total_timesteps:,}/{config.training_timesteps:,}\n"
                f"Avg Reward (10ep): {recent_stats['mean_reward']:.2f}\n"
                f"Avg Sharpe (10ep): {recent_stats['mean_sharpe']:.2f}\n"
                f"Avg WR (10ep): {recent_stats['mean_win_rate']:.2%}"
            )
        
        # === EVALUATION ===
        if total_timesteps - last_eval_time >= config.eval_frequency:
            eval_stats = evaluate_agent(agent, env, num_episodes=3)
            
            # Check for best model
            if eval_stats['mean_sharpe'] > best_sharpe:
                best_sharpe = eval_stats['mean_sharpe']
                best_model_path = os.path.join(config.checkpoint_dir, "best_model.pt")
                agent.save_model(best_model_path)
                log.info(f"🏆 New best Sharpe: {best_sharpe:.2f}")
                notifier.send_update(f"🏆 New best model! Sharpe: {best_sharpe:.2f}")
            
            last_eval_time = total_timesteps
        
        # === CHECKPOINTING ===
        if total_timesteps - last_save_time >= config.save_frequency:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"checkpoint_{total_timesteps}.pt"
            )
            agent.save_checkpoint(checkpoint_path)
            metrics.save(os.path.join(config.runs_dir, "metrics.json"))
            
            log.info(f"💾 Checkpoint saved: {checkpoint_path}")
            last_save_time = total_timesteps
        
        episode += 1
    
    # === TRAINING COMPLETE ===
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    
    log.info(f"\n{'='*70}")
    log.info(f"✅ Training {'interrupted' if SHUTDOWN_REQUESTED else 'complete'}!")
    log.info(f"  Total episodes: {episode}")
    log.info(f"  Total timesteps: {total_timesteps:,}")
    log.info(f"  Training time: {hours}h {minutes}m")
    log.info(f"{'='*70}\n")
    
    # Save final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "final_checkpoint.pt")
    agent.save_checkpoint(final_path)
    metrics.save(os.path.join(config.runs_dir, "final_metrics.json"))
    
    # Final evaluation
    log.info("🔍 Running final evaluation...")
    final_eval = evaluate_agent(agent, env, num_episodes=10)
    
    # Final notification
    notifier.send_update(
        f"✅ Training Complete!\n"
        f"Episodes: {episode}\n"
        f"Steps: {total_timesteps:,}\n"
        f"Time: {hours}h {minutes}m\n"
        f"Final Sharpe: {final_eval['mean_sharpe']:.2f}\n"
        f"Final WR: {final_eval['mean_win_rate']:.2%}"
    )
    
    return metrics, final_eval


def main():
    """Main entry point for advanced trading system."""
    parser = argparse.ArgumentParser(description="Advanced Trading Bot System")
    parser.add_argument('--config', type=str, help='Path to config JSON')
    parser.add_argument('--preset', type=str, 
                       choices=['development', 'production', 'testing'],
                       default='development',  # DEFAULT TO DEVELOPMENT
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
    
    # FORCE testnet for safety unless explicitly production
    if args.preset != 'production':
        config.use_testnet = True
        config.paper_trade = True
        log.warning("⚠️  Forcing TESTNET mode for safety (use --preset production for live)")
    
    config.print_summary()
    
    # Validate configuration
    valid, errors = config.validate()
    if not valid:
        log.error("❌ Configuration validation failed:")
        for error in errors:
            log.error(f"  • {error}")
        sys.exit(1)
    
    # Validate credentials match environment
    cred_valid, cred_msg = config.validate_credentials()
    if not cred_valid:
        log.error(f"❌ Credential validation failed: {cred_msg}")
        log.error("   Please check your .env file and OKX_IS_TESTNET setting")
        sys.exit(1)
    
    # Extra safety check for live trading
    if args.mode == 'live' and not config.paper_trade:
        log.critical("🔴 LIVE TRADING MODE ACTIVATED")
        log.critical("⚠️  This will execute REAL trades with REAL money!")
        
        response = input("Type 'I UNDERSTAND THE RISKS' to continue: ")
        if response != "I UNDERSTAND THE RISKS":
            log.info("👍 Smart choice - exiting safely")
            sys.exit(0)
    
    # Initialize system
    env, agent, logger, notifier, metrics = initialize_system(config)
    
    # Resume from checkpoint
    if args.resume:
        log.info(f"📂 Resuming from checkpoint: {args.resume}")
        agent.load_checkpoint(args.resume)
    
    # Evaluation mode
    if args.eval:
        log.info("🔍 Evaluation mode")
        eval_stats = evaluate_agent(agent, env, num_episodes=20)
        
        print("\n" + "="*70)
        print("📊 EVALUATION RESULTS")
        print("="*70)
        print(f"Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        print(f"Mean Return: {eval_stats['mean_return']:.2%}")
        print(f"Mean Sharpe: {eval_stats['mean_sharpe']:.2f}")
        print(f"Mean Win Rate: {eval_stats['mean_win_rate']:.2%}")
        print("="*70 + "\n")
        
        return
    
    # Training mode
    try:
        metrics, final_eval = train_loop(env, agent, logger, notifier, metrics, config)
        
        # Save final summary
        summary = {
            'config': config.to_dict(),
            'final_evaluation': final_eval,
            'training_metrics': metrics.get_recent_stats(100),
            'completed_at': datetime.now().isoformat()
        }
        
        summary_path = os.path.join(config.runs_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        log.info(f"📄 Training summary saved: {summary_path}")
    
    except Exception as e:
        log.exception(f"❌ Training failed with error: {e}")
        notifier.send_critical(f"Training crashed: {str(e)}")
        raise
    
    finally:
        log.info("👋 Shutting down...")


if __name__ == "__main__":
    main()