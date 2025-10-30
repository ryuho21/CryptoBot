"""
tuner.py

Optuna-based hyperparameter tuning for PPO. This script runs quick short trials
to find promising hyperparameters and writes best params to runs/best_params.json.

Usage:
    python tuner.py --trials 20 --timeout 1800
"""

import os
import json
import time
import argparse
import optuna
import numpy as np
from typing import Optional, Any, Dict, List, Tuple

# defensive imports (project modules)
try:
    from config import Config
    from data_manager import DataManager
    from trading_env import TradingEnv
    from ppo_agent import PPOAgent
    from ppo_buffer import PPOBuffer
except Exception as e:
    raise ImportError("tuner.py expects core project modules to be present") from e

# Logging helper
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tuner")

# A light-weight short training run used as the optuna objective.
def short_run_eval(config: Config, trial: optuna.Trial, seed: int = 0, episodes: int = 2) -> float:
    """
    Build a small environment/agent for a short run and return a scalar reward metric.
    Uses sampled hyperparams from trial.
    """
    # 1. Sample hyperparameters
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    ppo_epochs = trial.suggest_int("ppo_epochs", 2, 8)
    num_minibatches = trial.suggest_int("num_minibatches", 4, 16)
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [64, 128, 256])
    entropy_coef = trial.suggest_loguniform("entropy_coef", 1e-4, 1e-2)
    
    # 2. Update config object with trial params
    config.learning_rate = lr
    config.ppo_epochs = ppo_epochs
    config.num_minibatches = num_minibatches
    config.lstm_hidden_size = lstm_hidden_size
    config.entropy_coef = entropy_coef
    config.num_steps = 32 # Reduce rollout length for tuning speed
    
    # 3. Setup core components (same as main.py train_loop)
    dm = DataManager(config)
    envs = [TradingEnv(dm, config, seed=seed + i) for i in range(config.num_parallel_envs)]
    agent = PPOAgent(config, dm, envs)
    buffer = PPOBuffer(config, config.num_parallel_envs)
    
    # Initial states
    obs, _ = zip(*[env.reset() for env in envs])
    last_obs = np.stack(obs)
    # Using simple mock init_hidden_state (need access to main.py helper)
    def _init_h(cfg: Config, num_envs: int):
        H = cfg.lstm_hidden_size
        return np.zeros((cfg.rnn_layers, num_envs, H), dtype=np.float32)

    last_h = _init_h(config, config.num_parallel_envs)
    last_done = np.zeros(config.num_parallel_envs, dtype=np.bool_)
    
    total_rewards: List[float] = []
    
    # 4. Short Training Run
    for episode in range(episodes):
        for step in range(config.num_steps):
            # Rollout
            actions, values, log_probs, next_h = agent.get_action_and_value(last_obs, last_h)
            results = [env.step(actions[i]) for i, env in enumerate(envs)]
            next_obs_list, rewards, dones, infos = zip(*results)
            
            # Add rewards to tracking
            total_rewards.extend(rewards)

            # Update states/buffer
            next_obs = np.stack(next_obs_list)
            rewards = np.array(rewards, dtype=np.float32) 
            dones = np.array(dones, dtype=np.bool_) 
            
            for i in range(config.num_parallel_envs):
                buffer.add(i, last_obs[i], actions[i], rewards[i], values[i], log_probs[i], dones[i], last_h, next_h)
            
            last_obs = next_obs
            last_h = next_h
            last_done = dones
            buffer.ptr += 1
            
            for i, env in enumerate(envs):
                if dones[i]:
                    new_obs, _ = env.reset()
                    last_obs[i] = new_obs
                    # Simple h_state reset (for GRU/h_t)
                    if isinstance(last_h, np.ndarray):
                       last_h[:, i, :] = 0.0

        # PPO Update
        last_value, _ = agent.get_value(last_obs, last_h)
        buffer.compute_gae_and_returns(last_value.flatten(), last_done)
        agent.train(buffer)
        buffer.reset()

    # 5. Return metric (average reward)
    avg = float(np.mean(total_rewards))
    return avg

def objective(trial: optuna.Trial) -> float:
    cfg = Config()
    # use small trial seed from trial number to diversify runs
    seed = int(time.time() * 1000) % 10000 
    # Run 2 short episodes per trial
    score = short_run_eval(cfg, trial, seed=seed, episodes=2)
    return score

def run_study(n_trials: int = 20, timeout: Optional[int] = None, study_name: str = "ppo_tuning", storage: Optional[str] = None) -> Dict[str, Any]:
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name, storage=storage, load_if_exists=True)
    logger.info("Starting Optuna study (trials=%s timeout=%s)", n_trials, timeout)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    logger.info("Best trial: value=%s params=%s", study.best_value, study.best_params)
    
    # write best params
    os.makedirs("runs", exist_ok=True)
    with open("runs/best_params.json", "w") as f:
        json.dump({"value": study.best_value, "params": study.best_params}, f, indent=2)
        
    return study.best_params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=20, help='Number of trials for tuning.')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout in seconds for tuning.')
    args = parser.parse_args()
    
    run_study(n_trials=args.trials, timeout=args.timeout)