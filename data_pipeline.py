# data_pipeline.py - PRODUCTION-READY DATA PIPELINE
"""
Complete data pipeline with:
- CCXT integration
- Robust error handling
- Feature engineering
- Online normalization
- Mock data fallback
"""

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime
import time
from typing import Optional
import logging

from config import Config
from typing import Dict


logger = logging.getLogger(__name__)


class FeatureScaler:
    """Online feature normalization with running statistics."""
    
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.running_mean = np.zeros(feature_dim, dtype=np.float32)
        self.running_std = np.ones(feature_dim, dtype=np.float32)
        self.count = 0
        self.epsilon = 1e-8
    
    def update(self, data: np.ndarray):
        """Update running statistics with new data."""
        if len(data) == 0:
            return
        
        batch_mean = np.mean(data, axis=0)
        batch_std = np.std(data, axis=0)
        batch_count = len(data)
        
        if self.count == 0:
            self.running_mean = batch_mean
            self.running_std = np.maximum(batch_std, self.epsilon)
            self.count = batch_count
        else:
            # Welford's online algorithm
            new_count = self.count + batch_count
            delta = batch_mean - self.running_mean
            new_mean = self.running_mean + delta * batch_count / new_count
            
            m_a = self.running_std**2 * self.count
            m_b = batch_std**2 * batch_count
            M2 = m_a + m_b + (delta**2) * self.count * batch_count / new_count
            new_std = np.sqrt(M2 / new_count)
            
            self.running_mean = new_mean
            self.running_std = np.maximum(new_std, self.epsilon)
            self.count = new_count
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using running statistics."""
        return (data - self.running_mean) / self.running_std
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data."""
        return data * self.running_std + self.running_mean


class ExchangeDataSource:
    """
    Complete exchange data source with:
    - Real CCXT integration
    - Fallback mock data
    - Feature engineering
    - Online normalization
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = FeatureScaler(config.observation_shape[1])
        self.data_store = pd.DataFrame()
        
        # Initialize exchange
        self._init_exchange()
        
        # Feature columns
        self.feature_cols = ["open", "high", "low", "close", "volume", 
                             "ATR", "RSI", "MACD_Diff", "STOCH_K", "VPT"]
    
    def _init_exchange(self):
        """Initialize CCXT exchange with proper error handling."""
        try:
            exchange_config = {
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "password": self.config.api_passphrase,
                "enableRateLimit": True,
                "timeout": 30000,
            }
            
            self.exchange = getattr(ccxt, self.config.exchange_id)(exchange_config)
            
            if self.config.use_testnet:
                self.exchange.set_sandbox_mode(True)
                logger.info("✅ OKX Sandbox Mode Enabled (Data Pipeline)")
            else:
                logger.info("✅ OKX Live Mode Enabled (Data Pipeline)")
            
            # Test connection
            self.exchange.load_markets()
            logger.info(f"✅ Exchange initialized: {self.config.exchange_id}")
            
        except Exception as e:
            logger.warning(f"⚠️ Exchange initialization failed: {e}. Using mock data.")
            self.exchange = None
    
    def _fetch_historical(self, limit: int = 2000) -> pd.DataFrame:
        """Fetch historical OHLCV data with retry logic."""
        symbol = self.config.symbol
        timeframe = self.config.timeframe
        
        if self.exchange is None:
            return self._generate_mock_data(limit)
        
        logger.info(f"📈 Fetching {limit} {timeframe} candles for {symbol}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                
                if not ohlcv:
                    raise ValueError("Empty OHLCV response")
                
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("datetime", inplace=True)
                df.drop("timestamp", axis=1, inplace=True)
                
                logger.info(f"✅ Fetched {len(df)} candles")
                return df
                
            except ccxt.NetworkError as e:
                logger.warning(f"Network error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break
        
        logger.warning("⚠️ Falling back to mock data")
        return self._generate_mock_data(limit)
    
    def _generate_mock_data(self, length: int) -> pd.DataFrame:
        """Generate realistic synthetic OHLCV data."""
        logger.info(f"🎲 Generating {length} mock candles")
        
        # Generate realistic price series with trend and noise
        # FIX: Use 'min' instead of deprecated 'm' when used for minute frequency
        freq = self.config.timeframe.replace('m', 'min')
        idx = pd.date_range(end=pd.Timestamp.now(), periods=length, freq=freq)
        
        # Starting price
        base_price = 40000.0
        
        # Generate price with geometric brownian motion
        returns = np.random.normal(0.0001, 0.02, length) # drift and volatility
        cumulative_returns = np.exp(np.cumsum(returns))
        close_prices = base_price * cumulative_returns
        
        # Generate OHLC from close
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.005, length)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.005, length)))
        
        # Open prices (lag close by 1)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Volume with realistic distribution
        avg_volume = 1000
        volume = np.random.lognormal(np.log(avg_volume), 0.5, length)
        
        df = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume
        }, index=idx)
        
        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators as features."""
        df_ta = df.copy()
        
        try:
            import ta
            
            # Volatility
            df_ta["ATR"] = ta.volatility.AverageTrueRange(
                df_ta["high"], df_ta["low"], df_ta["close"], window=14
            ).average_true_range()
            
            # Momentum
            df_ta["RSI"] = ta.momentum.RSIIndicator(
                df_ta["close"], window=14
            ).rsi()
            
            # Trend
            macd = ta.trend.MACD(df_ta["close"])
            df_ta["MACD_Diff"] = macd.macd_diff()
            
            # Stochastic
            df_ta["STOCH_K"] = ta.momentum.StochasticOscillator(
                df_ta["high"], df_ta["low"], df_ta["close"]
            ).stoch()
            
            # Volume
            df_ta["VPT"] = ta.volume.volume_price_trend(df_ta["close"], df_ta["volume"])
            
            # FIX: Replace deprecated fillna(method='...') with bfill/ffill
            df_ta = df_ta.bfill()
            df_ta = df_ta.ffill()
            df_ta.fillna(0.0, inplace=True)
            
        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            # Fallback: use zeros
            for col in ["ATR", "RSI", "MACD_Diff", "STOCH_K", "VPT"]:
                if col not in df_ta.columns:
                    df_ta[col] = 0.0
        
        return df_ta[self.feature_cols]
    
    def load_initial_data(self):
        """Load and preprocess initial historical data."""
        raw_df = self._fetch_historical(limit=2000)
        processed_df = self._feature_engineering(raw_df)
        
        # Update scaler
        self.scaler.update(processed_df.values)
        
        # Store data (skip warmup period)
        warmup = max(self.config.window_size, 50)
        self.data_store = processed_df.iloc[warmup:].reset_index(drop=True)
        
        logger.info(f"✅ Data pipeline ready. Total steps: {len(self.data_store)}")
    
    def update_live_data(self) -> np.ndarray:
        """Simulate live data update (for live trading mode)."""
        if len(self.data_store) == 0:
            self.load_initial_data()
            return self.get_latest_observation()
        
        # Fetch latest candle
        try:
            if self.exchange:
                latest = self.exchange.fetch_ohlcv(
                    self.config.symbol, 
                    self.config.timeframe, 
                    limit=1
                )
                if latest:
                    new_row = pd.DataFrame(
                        latest, 
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    new_row["datetime"] = pd.to_datetime(new_row["timestamp"], unit="ms")
                    new_row.set_index("datetime", inplace=True)
                    new_row.drop("timestamp", axis=1, inplace=True)
                    
                    # Add features
                    combined = pd.concat([self.data_store.tail(50), new_row])
                    processed = self._feature_engineering(combined)
                    new_features = processed.iloc[-1:].values
                    
                    # Update scaler and store
                    self.scaler.update(new_features)
                    self.data_store = pd.concat([self.data_store, processed.iloc[-1:]])
                    
                    logger.debug("✅ Live data updated")
        
        except Exception as e:
            logger.warning(f"Live data update failed: {e}. Using last observation.")
        
        return self.get_latest_observation()
    
    def get_latest_observation(self) -> np.ndarray:
        """
        Get normalized window of latest market state.
        
        Returns:
            obs: (W, F) normalized features
        """
        window_raw = self.data_store.iloc[-self.config.window_size:].values
        
        # Pad if necessary
        if len(window_raw) < self.config.window_size:
            padding = np.zeros(
                (self.config.window_size - len(window_raw), self.scaler.feature_dim),
                dtype=np.float32
            )
            window_raw = np.concatenate([padding, window_raw], axis=0)
        
        # Normalize
        return self.scaler.transform(window_raw).astype(np.float32)
    
    def get_market_price(self, step_index: int) -> float:
        """Get close price at given step."""
        if step_index < 0 or step_index >= len(self.data_store):
            return self.data_store["close"].iloc[-1]
        return self.data_store["close"].iloc[step_index]
    
    def get_ohlcv_at_step(self, step_index: int) -> Dict:
        """Get full OHLCV at given step."""
        if step_index < 0 or step_index >= len(self.data_store):
            step_index = len(self.data_store) - 1
        
        row = self.data_store.iloc[step_index]
        return {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        }