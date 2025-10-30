import os
import time
import numpy as np
import pandas as pd
import ccxt
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta


class DataManager:
    def __init__(self, config):
        self.config = config
        self.cache_dir = os.path.join("runs", "data_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Features used in the model (10 features: 4 price, 1 return, 3 TA, 2 runtime)
        self.feat_cols = ["open", "high", "low", "close", "log_ret", "ema20", "ema50", "macd", "position", "equity"]

        # --- CCXT Initialization ---
        is_testnet = os.environ.get("OKX_IS_TESTNET", "False").lower() == "true"
        
        exchange_config = {
            "apiKey": os.environ.get("OKX_API_KEY", ""),
            "secret": os.environ.get("OKX_API_SECRET", ""),
            "password": os.environ.get("OKX_PASSPHRASE", ""),
            "enableRateLimit": True,
        }
        
        try:
            self.exchange = ccxt.okx(exchange_config)
            
            if is_testnet:
                # ✅ Critical: use sandbox mode for demo API keys
                self.exchange.set_sandbox_mode(True)
                print("INFO: Initialized OKX in TESTNET (sandbox) mode.")
            else:
                print("INFO: Initialized OKX in LIVE mode.")

        except Exception as e:
            print(f"CCXT Initialization Error: {e}")
            self.exchange = None

    # ======================================================
    # Load and cache OHLCV data
    # ======================================================
    def load_symbol(self, symbol: str, timeframe: str) -> pd.DataFrame:
        if not self.exchange:
            raise RuntimeError("CCXT Exchange failed to initialize.")
             
        # Path for cached file
        fname = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        fpath = os.path.join(self.cache_dir, fname)
        
        # Check cache validity
        cache_valid = False
        if os.path.exists(fpath):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if (datetime.now() - file_mtime) < timedelta(hours=self.config.cache_ttl_hours):
                cache_valid = True
                
        # Load data
        if cache_valid:
            print(f"INFO: Loading cached data for {symbol}/{timeframe}")
            df = pd.read_csv(fpath, parse_dates=["timestamp"])
        else:
            print(f"INFO: Fetching new data for {symbol}/{timeframe}...")
            df = self._fetch_ohlcv(symbol, timeframe)
            
            if not df.empty:
                df.to_csv(fpath, index=False)
                print(f"INFO: Fetched and cached new data. Rows: {len(df)}")
            else:
                print(f"WARNING: Fetched empty data for {symbol}/{timeframe}.")
                return pd.DataFrame()

        # Compute indicators
        df = self.compute_indicators(df)
        
        # Skip initial rows needed for warm-up
        df_start_idx = max(self.config.window_size, 50) 
        if len(df) < df_start_idx:
            raise ValueError(f"Not enough data ({len(df)} rows) fetched for training.")
            
        return df.iloc[df_start_idx:].reset_index(drop=True)

    def _fetch_ohlcv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch OHLCV from CCXT, handling pagination."""
        
        now = self.exchange.milliseconds()
        lookback_ms = self.config.lookback_minutes * 60 * 1000
        since = now - lookback_ms
        
        all_candles = []
        limit = 1000 
        
        while True:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
            except ccxt.ExchangeError as e:
                raise RuntimeError(f"CCXT Exchange Error during fetch: {e}")
            
            if not candles:
                break
            
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            
            time.sleep(self.exchange.rateLimit / 1000) 

            if len(candles) < limit:
                break

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset=['timestamp'], keep='first')
        return df

    # ======================================================
    # Feature Engineering
    # ======================================================
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Log returns
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)

        # Technical Indicators
        df["ema20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
        df["ema50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
        macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        
        # Fill NaNs
        df = df.bfill().ffill().fillna(0.0)
        
        # Runtime placeholders
        df['position'] = 0.0
        df['equity'] = 1.0

        return df
    
    # ======================================================
    # Return a windowed observation
    # ======================================================
    def windowed_features(self, df: pd.DataFrame, idx: int, window: int) -> np.ndarray:
        seg = df.iloc[max(0, idx - window + 1): idx + 1].copy()
        
        # Pad if needed
        if len(seg) < window:
            pad_rows = window - len(seg)
            padding = np.zeros((pad_rows, len(self.feat_cols)), dtype=np.float32)
            features = seg[self.feat_cols].values.astype(np.float32)
            obs = np.vstack([padding, features])
        else:
            obs = seg[self.feat_cols].values.astype(np.float32)

        # Normalize observation (skip last two runtime features)
        if self.config.normalize_obs:
            price_indicators = obs[:, :-2]
            mean = np.mean(price_indicators, axis=0)
            std = np.std(price_indicators, axis=0)
            std[std == 0] = 1.0
            obs[:, :-2] = (price_indicators - mean) / std
            
        return obs
