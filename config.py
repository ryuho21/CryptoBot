# config.py - PRODUCTION-READY CONFIGURATION SYSTEM
"""
Complete configuration management with:
- Environment variable validation
- Secure credential handling
- Configuration presets
- Type safety and validation
- Serialization with security
"""

import os
import json
import torch
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import hashlib
from datetime import datetime


@dataclass
class Config:
    """Production-ready configuration for trading bot."""
    
    # ========== EXCHANGE SETTINGS ==========
    exchange_id: str = 'okx'
    symbol: str = 'BTC/USDT:USDT'
    timeframe: str = '1m'
    use_testnet: bool = True
    cache_ttl_hours: int = 24
    lookback_minutes: int = 10080  # 7 days
    
    # Credentials (loaded from environment - NEVER hardcoded)
    api_key: str = field(default_factory=lambda: os.environ.get('OKX_API_KEY', ''))
    api_secret: str = field(default_factory=lambda: os.environ.get('OKX_API_SECRET', ''))
    api_passphrase: str = field(default_factory=lambda: os.environ.get('OKX_PASSPHRASE', ''))
    
    # ========== NETWORK ARCHITECTURE ==========
    rnn_type: str = 'LSTM'  # 'GRU' or 'LSTM'
    hidden_size: int = 256
    num_layers: int = 2
    feature_dim: int = 10  # Features per timestep
    window_size: int = 60  # Lookback window
    
    # Account state
    account_dim: int = 3  # [net_worth, position, volatility]
    action_dim: int = 3  # HOLD, BUY, SELL
    
    # ========== RL HYPERPARAMETERS ==========
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_beta: float = 0.01
    
    learning_rate_actor: float = 1e-4
    learning_rate_critic: float = 3e-4
    max_grad_norm: float = 0.5
    
    ppo_epochs: int = 4
    batch_size: int = 256
    rollout_steps: int = 2048
    num_parallel_envs: int = 1
    
    # ========== IQN CRITIC SETTINGS ==========
    iqn_n_quantiles: int = 8
    iqn_k_quantiles: int = 32  # For value estimation
    iqn_tau_soft_update: float = 0.005
    iqn_hidden_size: int = 64
    
    # ========== AUXILIARY LOSSES ==========
    wm_loss_weight: float = 0.1
    rnd_loss_weight: float = 0.01
    rnd_predictor_lr: float = 1e-4
    intrinsic_reward_scale: float = 0.01
    intrinsic_reward_clip: float = 1.0
    
    # ========== ENVIRONMENT SETTINGS ==========
    initial_balance: float = 100000.0
    commission_per_trade: float = 0.00075  # 0.075%
    normalize_obs: bool = True
    
    # ========== RISK MANAGEMENT ==========
    kelly_criterion_fraction: float = 0.25
    max_position_leverage: float = 3.0
    max_position_size: float = 0.25
    max_drawdown_pct: float = 0.15
    stop_loss_pct: float = 0.03  # 3%
    take_profit_pct: float = 0.06  # 6%
    
    trade_throttle_seconds: int = 5
    max_trades_per_minute: int = 5
    order_timeout_seconds: int = 30
    
    # ========== CURRICULUM LEARNING ==========
    curriculum_window: int = 100
    initial_volatility_factor: float = 0.5
    final_volatility_factor: float = 1.5
    curriculum_delta: float = 0.05
    
    # ========== TRAINING SETTINGS ==========
    training_timesteps: int = 5_000_000
    eval_frequency: int = 50000
    save_frequency: int = 100000
    max_retries: int = 3
    
    # ========== PAPER TRADING ==========
    paper_trade: bool = True
    spread_pct: float = 0.0005
    slippage_pct: float = 0.001
    
    # ========== LOGGING & MONITORING ==========
    runs_dir: str = 'runs'
    log_dir: str = 'logs'
    checkpoint_dir: str = 'runs/checkpoints'
    
    # Notifications
    telegram_token: str = field(default_factory=lambda: os.environ.get('TELEGRAM_TOKEN', ''))
    telegram_chat_id: str = field(default_factory=lambda: os.environ.get('TELEGRAM_CHAT_ID', ''))
    discord_webhook: str = field(default_factory=lambda: os.environ.get('DISCORD_WEBHOOK_URL', ''))
    
    # Dashboard
    dashboard_port: int = 8080
    enable_dashboard: bool = False
    
    # ========== COMPUTED PROPERTIES ==========
    @property
    def device(self) -> torch.device:
        """Auto-detect best available device."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @property
    def observation_shape(self) -> tuple:
        """Calculate observation space shape."""
        return (self.window_size, self.feature_dim)
    
    @property
    def has_credentials(self) -> bool:
        """Check if API credentials are set."""
        return bool(self.api_key and self.api_secret and self.api_passphrase)
    
    @property
    def has_notifications(self) -> bool:
        """Check if notification channels are configured."""
        return bool(self.telegram_token and self.telegram_chat_id) or bool(self.discord_webhook)
    
    @property
    def config_hash(self) -> str:
        """Generate hash of config for versioning."""
        config_str = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # ========== VALIDATION ==========
    def validate(self) -> tuple[bool, List[str]]:
        """Comprehensive configuration validation."""
        errors = []
        
        # Learning rates
        if not 0 < self.learning_rate_actor <= 0.1:
            errors.append(f"learning_rate_actor out of range: {self.learning_rate_actor}")
        if not 0 < self.learning_rate_critic <= 0.1:
            errors.append(f"learning_rate_critic out of range: {self.learning_rate_critic}")
        
        # Gamma and GAE
        if not 0 < self.gamma <= 1.0:
            errors.append(f"gamma must be in (0, 1]: {self.gamma}")
        if not 0 < self.gae_lambda <= 1.0:
            errors.append(f"gae_lambda must be in (0, 1]: {self.gae_lambda}")
        
        # Architecture
        if self.hidden_size < 32 or self.hidden_size > 2048:
            errors.append(f"hidden_size out of range [32, 2048]: {self.hidden_size}")
        if self.window_size < 10:
            errors.append(f"window_size too small: {self.window_size}")
        
        # Risk parameters
        if not 0 < self.max_drawdown_pct <= 0.5:
            errors.append(f"max_drawdown_pct out of range (0, 0.5]: {self.max_drawdown_pct}")
        if self.max_position_leverage < 1.0:
            errors.append(f"max_position_leverage must be >= 1.0: {self.max_position_leverage}")
        
        # Batch settings
        if self.batch_size > self.rollout_steps:
            errors.append(f"batch_size ({self.batch_size}) > rollout_steps ({self.rollout_steps})")
        if self.rollout_steps % self.batch_size != 0:
            errors.append(f"rollout_steps not divisible by batch_size")
        
        # Credentials (warning only in production)
        if not self.use_testnet and not self.has_credentials:
            errors.append("Production mode requires API credentials")
        
        return len(errors) == 0, errors
    def validate_credentials(self) -> Tuple[bool, str]:
        """
        Validate API credentials match environment mode.
        
        Returns:
            (is_valid, message)
        """
        if not self.has_credentials:
            if not self.use_testnet and not self.paper_trade:
                return False, "Live trading requires API credentials"
            return True, "Paper trading mode - no credentials needed"
        
        # Check if credentials look like testnet keys
        is_testnet_key = any([
            'demo' in self.api_key.lower(),
            'test' in self.api_key.lower(),
            'sandbox' in self.api_key.lower(),
        ])
        
        # Validate match
        if self.use_testnet and not is_testnet_key:
            return False, "Testnet mode enabled but credentials don't look like testnet keys"
        
        if not self.use_testnet and is_testnet_key:
            return False, "⚠️ DANGER: Live mode with testnet credentials - this will fail"
        
        return True, "Credentials validated"
    # ========== PRESETS ==========
    @classmethod
    def development(cls) -> 'Config':
        """Fast iteration preset."""
        config = cls()
        config.use_testnet = True
        config.rollout_steps = 512
        config.batch_size = 128
        config.training_timesteps = 100000
        config.hidden_size = 128
        config.num_layers = 1
        config.ppo_epochs = 2
        config.save_frequency = 10000
        return config
    
    @classmethod
    def production(cls) -> 'Config':
        """Live trading optimized preset."""
        config = cls()
        config.use_testnet = False
        config.paper_trade = False
        config.rollout_steps = 4096
        config.batch_size = 256
        config.hidden_size = 512
        config.num_layers = 3
        config.ppo_epochs = 6
        config.max_position_leverage = 2.0
        config.kelly_criterion_fraction = 0.1
        config.enable_dashboard = True
        return config
    
    @classmethod
    def testing(cls) -> 'Config':
        """Minimal resources for testing."""
        config = cls()
        config.use_testnet = True
        config.rollout_steps = 128
        config.batch_size = 32
        config.training_timesteps = 5000
        config.hidden_size = 64
        config.num_layers = 1
        config.ppo_epochs = 1
        return config
    
    # ========== SERIALIZATION ==========
    def save(self, path: str):
        """Save config with credential masking."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        data = asdict(self)
        # Mask sensitive fields
        sensitive = ['api_key', 'api_secret', 'api_passphrase', 'telegram_token', 
                    'telegram_chat_id', 'discord_webhook']
        for key in sensitive:
            if key in data and data[key]:
                data[key] = "***REDACTED***"
        
        # Add metadata
        data['_saved_at'] = datetime.now().isoformat()
        data['_config_hash'] = self.config_hash
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Config saved: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load config (credentials from env)."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Remove redacted credentials
        sensitive = ['api_key', 'api_secret', 'api_passphrase', 'telegram_token',
                    'telegram_chat_id', 'discord_webhook', '_saved_at', '_config_hash']
        for key in sensitive:
            data.pop(key, None)
        
        config = cls(**data)
        print(f"✅ Config loaded: {path}")
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def update(self, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")
    
    # ========== DISPLAY ==========
    def print_summary(self):
        """Print formatted configuration summary."""
        print("\n" + "="*70)
        print("🤖 TRADING BOT CONFIGURATION")
        print("="*70)
        
        print(f"\n📊 EXCHANGE:")
        print(f"  • Exchange: {self.exchange_id.upper()}")
        print(f"  • Symbol: {self.symbol}")
        print(f"  • Timeframe: {self.timeframe}")
        print(f"  • Mode: {'🧪 TESTNET' if self.use_testnet else '🔴 LIVE'}")
        print(f"  • Paper Trade: {'✅ Yes' if self.paper_trade else '❌ No'}")
        print(f"  • Credentials: {'✅ Set' if self.has_credentials else '⚠️  Missing'}")
        
        # Validate credentials
        cred_valid, cred_msg = self.validate_credentials()
        if not cred_valid:
            print(f"  • ⚠️  WARNING: {cred_msg}")
        
        print(f"\n🧠 NETWORK ARCHITECTURE:")
        print(f"  • RNN Type: {self.rnn_type}")
        print(f"  • Hidden Size: {self.hidden_size}")
        print(f"  • Num Layers: {self.num_layers}")
        print(f"  • Window: {self.window_size} steps")
        print(f"  • Features: {self.feature_dim}")
        
        print(f"\n🎯 TRAINING:")
        print(f"  • Total Steps: {self.training_timesteps:,}")
        print(f"  • Rollout: {self.rollout_steps}")
        print(f"  • Batch Size: {self.batch_size}")
        print(f"  • PPO Epochs: {self.ppo_epochs}")
        print(f"  • LR Actor: {self.learning_rate_actor:.2e}")
        print(f"  • LR Critic: {self.learning_rate_critic:.2e}")
        
        print(f"\n⚠️ RISK MANAGEMENT:")
        print(f"  • Balance: ${self.initial_balance:,.2f}")
        print(f"  • Max Leverage: {self.max_position_leverage}x")
        print(f"  • Max Drawdown: {self.max_drawdown_pct*100:.1f}%")
        print(f"  • Kelly Fraction: {self.kelly_criterion_fraction:.2f}")
        print(f"  • Commission: {self.commission_per_trade*100:.3f}%")
        print(f"  • Stop Loss: {self.stop_loss_pct*100:.1f}%")
        print(f"  • Take Profit: {self.take_profit_pct*100:.1f}%")
        
        print(f"\n📢 NOTIFICATIONS:")
        print(f"  • Telegram: {'✅ Configured' if self.telegram_token else '❌ Not set'}")
        print(f"  • Discord: {'✅ Configured' if self.discord_webhook else '❌ Not set'}")
        
        print(f"\n💻 SYSTEM:")
        print(f"  • Device: {self.device}")
        print(f"  • Config Hash: {self.config_hash}")
        print(f"  • Checkpoint Dir: {self.checkpoint_dir}")
        
        valid, errors = self.validate()
        if valid:
            print("\n✅ Configuration valid")
        else:
            print("\n❌ Configuration errors:")
            for err in errors:
                print(f"  • {err}")
        
        print("="*70 + "\n")


# ========== HELPER FUNCTIONS ==========
def get_config(preset: Optional[str] = None, config_path: Optional[str] = None) -> Config:
    """
    Get configuration with priority:
    1. config_path (if provided)
    2. preset (if provided)
    3. Environment variables
    4. Default
    """
    if config_path and os.path.exists(config_path):
        return Config.load(config_path)
    
    if preset:
        preset_map = {
            'development': Config.development,
            'dev': Config.development,
            'production': Config.production,
            'prod': Config.production,
            'testing': Config.testing,
            'test': Config.testing,
        }
        factory = preset_map.get(preset.lower())
        if factory:
            return factory()
        raise ValueError(f"Unknown preset: {preset}")
    
    return Config()


# ========== MAIN ==========
if __name__ == '__main__':
    import sys
    
    preset = sys.argv[1] if len(sys.argv) > 1 else None
    config = get_config(preset=preset)
    config.print_summary()
    
    valid, errors = config.validate()
    if not valid:
        print(f"\n❌ Validation failed:")
        for err in errors:
            print(f"  • {err}")
        sys.exit(1)
    
    # Save example
    config.save('runs/config_example.json')
    print("\n✅ Configuration validated and saved")
