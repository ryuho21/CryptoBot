# order_book_analyzer.py - ORDER BOOK DEPTH ANALYSIS
"""
Order book analysis for:
- Liquidity measurement
- Support/resistance detection
- Order flow imbalance
- Market microstructure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class OrderBookSnapshot:
    """Single order book snapshot."""
    
    def __init__(self, timestamp: float, bids: List[Tuple[float, float]], 
                 asks: List[Tuple[float, float]]):
        self.timestamp = timestamp
        self.bids = bids  # [(price, size), ...]
        self.asks = asks
        
        # Calculate mid price
        self.best_bid = bids[0][0] if bids else 0
        self.best_ask = asks[0][0] if asks else 0
        self.mid_price = (self.best_bid + self.best_ask) / 2 if bids and asks else 0
        
        # Calculate spread
        self.spread = self.best_ask - self.best_bid if bids and asks else 0
        self.spread_bps = (self.spread / self.mid_price) * 10000 if self.mid_price > 0 else 0


class OrderBookAnalyzer:
    """
    Analyze order book for trading signals.
    
    Features:
    - Liquidity analysis
    - Imbalance detection
    - Volume profile
    - Support/resistance
    """
    
    def __init__(self, config, max_history: int = 1000):
        self.config = config
        self.max_history = max_history
        self.snapshots = deque(maxlen=max_history)
    
    def add_snapshot(self, timestamp: float, 
                    bids: List[Tuple[float, float]],
                    asks: List[Tuple[float, float]]):
        """Add order book snapshot."""
        snapshot = OrderBookSnapshot(timestamp, bids, asks)
        self.snapshots.append(snapshot)
    
    def calculate_order_flow_imbalance(self, depth_levels: int = 10) -> float:
        """
        Calculate order flow imbalance.
        
        Args:
            depth_levels: number of levels to consider
        
        Returns:
            imbalance: positive = buy pressure, negative = sell pressure
        """
        if not self.snapshots:
            return 0.0
        
        snapshot = self.snapshots[-1]
        
        # Sum volumes at top N levels
        bid_volume = sum(size for _, size in snapshot.bids[:depth_levels])
        ask_volume = sum(size for _, size in snapshot.asks[:depth_levels])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        imbalance = (bid_volume - ask_volume) / total_volume
        return float(imbalance)
    
    def detect_support_resistance(self, price_range_pct: float = 0.02) -> Dict:
        """
        Detect support and resistance from order book.
        
        Args:
            price_range_pct: price range to analyze (as fraction)
        
        Returns:
            dict with support/resistance levels
        """
        if not self.snapshots:
            return {}
        
        snapshot = self.snapshots[-1]
        mid_price = snapshot.mid_price
        
        # Price range
        price_min = mid_price * (1 - price_range_pct)
        price_max = mid_price * (1 + price_range_pct)
        
        # Aggregate volume at price levels
        bid_volumes = {}
        for price, size in snapshot.bids:
            if price_min <= price <= price_max:
                bid_volumes[price] = bid_volumes.get(price, 0) + size
        
        ask_volumes = {}
        for price, size in snapshot.asks:
            if price_min <= price <= price_max:
                ask_volumes[price] = ask_volumes.get(price, 0) + size
        
        # Find strongest support (max bid volume)
        support_level = max(bid_volumes.items(), key=lambda x: x[1])[0] if bid_volumes else mid_price
        
        # Find strongest resistance (max ask volume)
        resistance_level = max(ask_volumes.items(), key=lambda x: x[1])[0] if ask_volumes else mid_price
        
        return {
            'support': support_level,
            'resistance': resistance_level,
            'support_volume': bid_volumes.get(support_level, 0),
            'resistance_volume': ask_volumes.get(resistance_level, 0),
            'mid_price': mid_price
        }
    
    def calculate_liquidity_score(self, depth_levels: int = 20) -> float:
        """
        Calculate market liquidity score.
        
        Args:
            depth_levels: levels to consider
        
        Returns:
            liquidity_score: higher = more liquid
        """
        if not self.snapshots:
            return 0.0
        
        snapshot = self.snapshots[-1]
        
        # Total volume in top N levels
        bid_volume = sum(size for _, size in snapshot.bids[:depth_levels])
        ask_volume = sum(size for _, size in snapshot.asks[:depth_levels])
        total_volume = bid_volume + ask_volume
        
        # Spread penalty
        spread_penalty = 1.0 / (1.0 + snapshot.spread_bps / 10)
        
        # Liquidity score
        liquidity = total_volume * spread_penalty
        
        return float(liquidity)
    
    def get_volume_weighted_prices(self, depth_levels: int = 10) -> Dict[str, float]:
        """
        Calculate volume-weighted bid/ask prices.
        
        Args:
            depth_levels: levels to consider
        
        Returns:
            dict with VWAP bid and ask
        """
        if not self.snapshots:
            return {'vwap_bid': 0.0, 'vwap_ask': 0.0}
        
        snapshot = self.snapshots[-1]
        
        # VWAP bid
        bid_sum = sum(price * size for price, size in snapshot.bids[:depth_levels])
        bid_vol = sum(size for _, size in snapshot.bids[:depth_levels])
        vwap_bid = bid_sum / bid_vol if bid_vol > 0 else 0.0
        
        # VWAP ask
        ask_sum = sum(price * size for price, size in snapshot.asks[:depth_levels])
        ask_vol = sum(size for _, size in snapshot.asks[:depth_levels])
        vwap_ask = ask_sum / ask_vol if ask_vol > 0 else 0.0
        
        return {
            'vwap_bid': float(vwap_bid),
            'vwap_ask': float(vwap_ask),
            'vwap_mid': float((vwap_bid + vwap_ask) / 2) if vwap_bid > 0 and vwap_ask > 0 else 0.0
        }
    
    def detect_market_pressure(self) -> str:
        """
        Detect market pressure direction.
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        imbalance = self.calculate_order_flow_imbalance()
        
        if imbalance > 0.1:
            return 'bullish'
        elif imbalance < -0.1:
            return 'bearish'
        else:
            return 'neutral'
    
    def get_analytics_summary(self) -> Dict:
        """Get comprehensive order book analytics."""
        if not self.snapshots:
            return {}
        
        snapshot = self.snapshots[-1]
        
        return {
            'timestamp': snapshot.timestamp,
            'mid_price': snapshot.mid_price,
            'spread': snapshot.spread,
            'spread_bps': snapshot.spread_bps,
            'imbalance': self.calculate_order_flow_imbalance(),
            'liquidity_score': self.calculate_liquidity_score(),
            'market_pressure': self.detect_market_pressure(),
            'support_resistance': self.detect_support_resistance(),
            'vwap_prices': self.get_volume_weighted_prices()
        }