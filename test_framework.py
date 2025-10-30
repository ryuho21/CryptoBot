# test_framework.py - PRODUCTION-READY TESTING FRAMEWORK
"""
Complete testing suite with:
- Unit tests for all components
- Integration tests
- Mock fixtures
- Performance benchmarks
- Coverage reporting
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

# Import all modules to test
from config import Config, get_config
from model_core import PolicyIQNNet, WorldModel, RNDNet, init_hidden_state
from ppo_buffer import PPOBuffer
from ppo_agent import PPOAgent, RunningMeanStd
from data_pipeline import ExchangeDataSource, FeatureScaler
from exchange_manager import RiskManager, ExchangeManager
from trading_env import TradingBotEnv
from curriculum_scheduler import CurriculumScheduler
from notification_service import NotificationService
from utils import (sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
                   calculate_metrics, exponential_moving_average, normalize_array)


# ==================== FIXTURES ====================

@pytest.fixture
def config():
    """Create test configuration."""
    cfg = Config.testing()
    cfg.use_testnet = True
    cfg.paper_trade = True
    return cfg


@pytest.fixture
def mock_observation_space():
    """Mock observation space for testing."""
    from gymnasium import spaces
    return {
        "features": spaces.Box(low=-np.inf, high=np.inf, shape=(60, 10), dtype=np.float32),
        "account": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    }


@pytest.fixture
def mock_action_space():
    """Mock action space for testing."""
    from gymnasium import spaces
    return spaces.Discrete(3)


@pytest.fixture
def sample_obs(config):
    """Generate sample observation."""
    return {
        "features": np.random.randn(config.window_size, config.feature_dim).astype(np.float32),
        "account": np.array([1.0, 0.0, 1.0], dtype=np.float32)
    }


@pytest.fixture
def mock_data_source(config):
    """Create mock data source."""
    import pandas as pd
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(40000, 41000, 1000),
        'high': np.random.uniform(40000, 41000, 1000),
        'low': np.random.uniform(40000, 41000, 1000),
        'close': np.random.uniform(40000, 41000, 1000),
        'volume': np.random.uniform(100, 500, 1000),
        'ATR': np.random.uniform(100, 200, 1000),
        'RSI': np.random.uniform(30, 70, 1000),
        'MACD_Diff': np.random.uniform(-50, 50, 1000),
        'STOCH_K': np.random.uniform(20, 80, 1000),
        'VPT': np.random.uniform(-1000, 1000, 1000)
    }, index=dates)
    
    source = Mock()
    source.data_store = data
    source.config = config
    source.get_latest_observation = Mock(return_value=np.random.randn(60, 10).astype(np.float32))
    source.get_market_price = Mock(return_value=40500.0)
    source.get_ohlcv_at_step = Mock(return_value={
        'open': 40000, 'high': 41000, 'low': 39500, 'close': 40500, 'volume': 300
    })
    
    return source


# ==================== UNIT TESTS ====================

class TestConfig:
    """Test configuration system."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        cfg = Config()
        assert cfg.symbol == 'BTC/USDT:USDT'
        assert cfg.action_dim == 3
        assert cfg.feature_dim == 10
    
    def test_config_validation(self):
        """Test config validation."""
        cfg = Config.testing()
        valid, errors = cfg.validate()
        assert valid
        assert len(errors) == 0
    
    def test_config_presets(self):
        """Test configuration presets."""
        dev = Config.development()
        assert dev.rollout_steps == 512
        
        prod = Config.production()
        assert prod.rollout_steps == 4096
        
        test = Config.testing()
        assert test.rollout_steps == 128
    
    def test_config_save_load(self, tmp_path):
        """Test config serialization."""
        cfg = Config.testing()
        path = tmp_path / "test_config.json"
        
        cfg.save(str(path))
        assert path.exists()
        
        loaded = Config.load(str(path))
        assert loaded.symbol == cfg.symbol
        assert loaded.rollout_steps == cfg.rollout_steps


class TestModelCore:
    """Test neural network architectures."""
    
    def test_policy_iqn_net_creation(self, config, mock_observation_space, mock_action_space):
        """Test PolicyIQNNet initialization."""
        model = PolicyIQNNet(mock_observation_space, mock_action_space, config)
        assert model is not None
        assert model.hidden_size == config.hidden_size
    
    def test_policy_forward_pass(self, config, mock_observation_space, mock_action_space, sample_obs):
        """Test forward pass."""
        device = torch.device('cpu')
        model = PolicyIQNNet(mock_observation_space, mock_action_space, config).to(device)
        
        obs_t = {
            "features": torch.from_numpy(sample_obs["features"]).unsqueeze(0).to(device),
            "account": torch.from_numpy(sample_obs["account"]).unsqueeze(0).to(device)
        }
        h_in = init_hidden_state(config, 1, device)
        
        action_logits, q_quantiles, h_out = model(obs_t, h_in, taus=torch.rand(1, 8, device=device))
        
        assert action_logits.shape == (1, 3)
        assert q_quantiles.shape == (1, 8)
        assert len(h_out) == 2
    
    def test_world_model_prediction(self, config, sample_obs):
        """Test world model."""
        device = torch.device('cpu')
        model = WorldModel(config).to(device)
        
        obs_t = {
            "features": torch.from_numpy(sample_obs["features"]).unsqueeze(0).to(device),
            "account": torch.from_numpy(sample_obs["account"]).unsqueeze(0).to(device)
        }
        action = torch.tensor([1], device=device)
        
        prediction = model(obs_t, action)
        expected_size = config.feature_dim * config.window_size + 3 + 1
        assert prediction.shape == (1, expected_size)
    
    def test_rnd_network(self, config):
        """Test RND network."""
        device = torch.device('cpu')
        model = RNDNet(config).to(device)
        
        features = torch.randn(4, config.feature_dim, device=device)
        target, predictor = model(features)
        
        assert target.shape == predictor.shape
        assert target.requires_grad == False
        assert predictor.requires_grad == True


class TestPPOBuffer:
    """Test PPO rollout buffer."""
    
    def test_buffer_creation(self, config):
        """Test buffer initialization."""
        buffer = PPOBuffer(config)
        assert buffer.rollout_steps == config.rollout_steps
        assert len(buffer) == 0
    
    def test_buffer_add(self, config, sample_obs):
        """Test adding transitions."""
        buffer = PPOBuffer(config)
        device = torch.device('cpu')
        h_in = init_hidden_state(config, 1, device)
        
        next_feat = sample_obs["features"][-1, :]
        
        buffer.add(
            sample_obs, next_feat, action=1, log_prob=0.5,
            reward=1.0, done=False, h_in=h_in,
            intrinsic_reward=0.1, value=0.8
        )
        
        assert len(buffer) == 1
    
    def test_buffer_full(self, config, sample_obs):
        """Test buffer full condition."""
        buffer = PPOBuffer(config)
        device = torch.device('cpu')
        h_in = init_hidden_state(config, 1, device)
        next_feat = sample_obs["features"][-1, :]
        
        for i in range(config.rollout_steps):
            buffer.add(sample_obs, next_feat, 1, 0.5, 1.0, False, h_in, 0.1, 0.8)
        
        assert buffer.is_full()
    
    def test_gae_computation(self, config, sample_obs):
        """Test GAE computation."""
        buffer = PPOBuffer(config)
        device = torch.device('cpu')
        h_in = init_hidden_state(config, 1, device)
        next_feat = sample_obs["features"][-1, :]
        
        for i in range(config.rollout_steps):
            buffer.add(sample_obs, next_feat, 1, 0.5, 1.0, False, h_in, 0.1, 0.8)
        
        buffer.compute_gae_and_returns(next_value=0.5, next_done=False)
        
        data = buffer.get()
        assert 'advantages' in data
        assert 'returns' in data


class TestRiskManager:
    """Test risk management."""
    
    def test_risk_manager_creation(self, config):
        """Test initialization."""
        rm = RiskManager(config)
        assert rm.net_worth == config.initial_balance
    
    def test_position_size_calculation(self, config):
        """Test Kelly criterion position sizing."""
        rm = RiskManager(config)
        size = rm.calculate_position_size_usd(40000.0, 0.0, 0.15)
        
        assert size >= 0
        assert size <= rm.net_worth * config.max_position_leverage
    
    def test_circuit_breaker(self, config):
        """Test circuit breaker triggering."""
        rm = RiskManager(config)
        
        # Trigger drawdown
        new_worth = config.initial_balance * (1 - config.max_drawdown_pct - 0.01)
        rm.update_metrics(new_worth)
        
        assert rm.circuit_breaker_tripped
    
    def test_trade_throttling(self, config):
        """Test trade rate limiting."""
        rm = RiskManager(config)
        
        # First trade should be allowed
        allowed, _ = rm.check_trade_throttling()
        assert allowed
        
        # Log trade
        rm.log_trade()
        
        # Immediate next trade should be throttled
        allowed, reason = rm.check_trade_throttling()
        assert not allowed


class TestTradingEnv:
    """Test trading environment."""
    
    def test_env_creation(self, config, mock_data_source):
        """Test environment initialization."""
        rm = RiskManager(config)
        em = ExchangeManager(rm, config)
        env = TradingBotEnv(mock_data_source, em, config)
        
        assert env is not None
        assert env.action_space.n == 3
    
    def test_env_reset(self, config, mock_data_source):
        """Test environment reset."""
        rm = RiskManager(config)
        em = ExchangeManager(rm, config)
        env = TradingBotEnv(mock_data_source, em, config)
        
        obs, info = env.reset()
        
        assert "features" in obs
        assert "account" in obs
        assert "net_worth" in info
    
    def test_env_step(self, config, mock_data_source):
        """Test environment step."""
        rm = RiskManager(config)
        em = ExchangeManager(rm, config)
        env = TradingBotEnv(mock_data_source, em, config)
        
        obs, _ = env.reset()
        next_obs, reward, terminated, truncated, info = env.step(0)  # HOLD
        
        assert "features" in next_obs
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)


class TestUtilities:
    """Test utility functions."""
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = sharpe_ratio(returns)
        assert isinstance(sharpe, float)
    
    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.01])
        sortino = sortino_ratio(returns)
        assert isinstance(sortino, float)
    
    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        equity = np.array([100, 110, 105, 115, 108, 120])
        dd = max_drawdown(equity)
        assert 0 <= dd <= 1
    
    def test_normalize_array(self):
        """Test array normalization."""
        arr = np.random.randn(100)
        
        normalized = normalize_array(arr, method='zscore')
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1


class TestCurriculumScheduler:
    """Test curriculum learning."""
    
    def test_curriculum_creation(self, config):
        """Test scheduler initialization."""
        scheduler = CurriculumScheduler(config)
        assert scheduler is not None
    
    def test_curriculum_update(self, config):
        """Test curriculum progression."""
        scheduler = CurriculumScheduler(config)
        
        # Good performance
        for i in range(20):
            scheduler.update({
                'episode_reward': 100,
                'sharpe': 1.5,
                'win_rate': 0.65
            })
        
        vol = scheduler.get_volatility_factor()
        assert vol >= 1.0  # Should increase difficulty


# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Integration tests for full system."""
    
    def test_full_training_step(self, config, mock_data_source):
        """Test complete training step."""
        rm = RiskManager(config)
        em = ExchangeManager(rm, config)
        env = TradingBotEnv(mock_data_source, em, config)
        agent = PPOAgent(config, env)
        
        # Reset
        obs, _ = env.reset()
        agent.reset_recurrent_state()
        
        # Collect rollout
        for _ in range(config.rollout_steps):
            h_in = tuple(t.clone() for t in agent.h_state)
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, _, _ = env.step(action)
            
            next_feat = next_obs["features"][-1, :]
            intrinsic = agent.compute_intrinsic_reward(next_feat)
            
            agent.buffer.add(obs, next_feat, action, log_prob, reward, done, h_in, intrinsic, value)
            
            obs = next_obs
            
            if done:
                break
        
        # Learn
        if agent.buffer.is_full():
            losses = agent.learn(obs, False)
            assert len(losses) == 5  # total, policy, critic, wm, rnd
    
    def test_checkpoint_save_load(self, config, mock_data_source, tmp_path):
        """Test checkpoint persistence."""
        rm = RiskManager(config)
        em = ExchangeManager(rm, config)
        env = TradingBotEnv(mock_data_source, em, config)
        agent = PPOAgent(config, env)
        
        # Save
        path = tmp_path / "test_checkpoint.pt"
        agent.save_checkpoint(str(path))
        assert path.exists()
        
        # Load
        agent2 = PPOAgent(config, env)
        agent2.load_checkpoint(str(path))
        
        # Compare states
        for p1, p2 in zip(agent.policy.parameters(), agent2.policy.parameters()):
            assert torch.allclose(p1, p2)


# ==================== PERFORMANCE BENCHMARKS ====================

class TestPerformance:
    """Performance and speed benchmarks."""
    
    def test_forward_pass_speed(self, config, mock_observation_space, mock_action_space, sample_obs, benchmark):
        """Benchmark forward pass speed."""
        device = torch.device('cpu')
        model = PolicyIQNNet(mock_observation_space, mock_action_space, config).to(device)
        
        obs_t = {
            "features": torch.from_numpy(sample_obs["features"]).unsqueeze(0).to(device),
            "account": torch.from_numpy(sample_obs["account"]).unsqueeze(0).to(device)
        }
        h_in = init_hidden_state(config, 1, device)
        
        def forward():
            with torch.no_grad():
                model(obs_t, h_in)
        
        benchmark(forward)
    
    def test_buffer_throughput(self, config, sample_obs, benchmark):
        """Benchmark buffer operations."""
        buffer = PPOBuffer(config)
        device = torch.device('cpu')
        h_in = init_hidden_state(config, 1, device)
        next_feat = sample_obs["features"][-1, :]
        
        def add_transition():
            buffer.add(sample_obs, next_feat, 1, 0.5, 1.0, False, h_in, 0.1, 0.8)
            if buffer.is_full():
                buffer.reset()
        
        benchmark(add_transition)


# ==================== RUN TESTS ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=.", "--cov-report=html"])