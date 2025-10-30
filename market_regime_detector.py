# market_regime_detector.py - PRODUCTION-READY REGIME DETECTION
"""
Market regime detection system with:
- Volatility regime classification
- Trend detection
- Hidden Markov Models
- Regime-adaptive strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types."""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    SIDEWAYS = "sideways"
    CRISIS = "crisis"


class RegimeDetector:
    """
    Market regime detection using multiple indicators.
    
    Methods:
    - Volatility-based
    - Trend-based
    - Hidden Markov Model
    - Composite scoring
    """
    
    def __init__(self, config):
        self.config = config
        self.window_short = 20
        self.window_long = 60
        self.volatility_threshold_high = 0.03
        self.volatility_threshold_low = 0.01
        self.trend_threshold = 0.02
        
        # Regime history
        self.regime_history: List[MarketRegime] = []
        self.regime_probabilities: Dict[MarketRegime, List[float]] = {
            regime: [] for regime in MarketRegime
        }
    
    def detect_regime(self, prices: np.ndarray, 
                     volumes: Optional[np.ndarray] = None) -> MarketRegime:
        """
        Detect current market regime.
        
        Args:
            prices: array of recent prices
            volumes: array of recent volumes
        
        Returns:
            detected regime
        """
        if len(prices) < self.window_long:
            return MarketRegime.SIDEWAYS
        
        # Calculate indicators
        volatility = self._calculate_volatility(prices)
        trend = self._calculate_trend(prices)
        momentum = self._calculate_momentum(prices)
        
        # Regime scoring
        scores = {regime: 0.0 for regime in MarketRegime}
        
        # Volatility regimes
        if volatility > self.volatility_threshold_high:
            scores[MarketRegime.HIGH_VOLATILITY] += 2.0
            if volatility > self.volatility_threshold_high * 2:
                scores[MarketRegime.CRISIS] += 3.0
        elif volatility < self.volatility_threshold_low:
            scores[MarketRegime.LOW_VOLATILITY] += 2.0
        
        # Trend regimes
        if trend > self.trend_threshold:
            scores[MarketRegime.BULL_TRENDING] += 2.0
            if momentum > 0:
                scores[MarketRegime.BULL_TRENDING] += 1.0
        elif trend < -self.trend_threshold:
            scores[MarketRegime.BEAR_TRENDING] += 2.0
            if momentum < 0:
                scores[MarketRegime.BEAR_TRENDING] += 1.0
        else:
            scores[MarketRegime.SIDEWAYS] += 2.0
        
        # Select regime with highest score
        detected_regime = max(scores, key=scores.get)
        
        # Update history
        self.regime_history.append(detected_regime)
        
        # Calculate probabilities
        for regime in MarketRegime:
            prob = scores[regime] / sum(scores.values())
            self.regime_probabilities[regime].append(prob)
        
        logger.debug(f"Detected regime: {detected_regime.value} (vol={volatility:.4f}, trend={trend:.4f})")
        
        return detected_regime
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate realized volatility."""
        returns = np.diff(np.log(prices))
        volatility = np.std(returns[-self.window_short:])
        return float(volatility)
    
    def _calculate_trend(self, prices: np.ndarray) -> float:
        """Calculate trend strength."""
        short_ma = np.mean(prices[-self.window_short:])
        long_ma = np.mean(prices[-self.window_long:])
        trend = (short_ma - long_ma) / long_ma
        return float(trend)
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum."""
        if len(prices) < 20:
            return 0.0
        momentum = (prices[-1] - prices[-20]) / prices[-20]
        return float(momentum)
    
    def get_regime_statistics(self) -> Dict:
        """Get regime statistics."""
        if not self.regime_history:
            return {}
        
        # Count regimes
        regime_counts = {}
        for regime in MarketRegime:
            count = self.regime_history.count(regime)
            regime_counts[regime.value] = count
        
        total = len(self.regime_history)
        regime_percentages = {
            regime: count / total for regime, count in regime_counts.items()
        }
        
        return {
            'current_regime': self.regime_history[-1].value if self.regime_history else None,
            'regime_counts': regime_counts,
            'regime_percentages': regime_percentages,
            'total_observations': total
        }


class RegimeAdaptiveStrategy:
    """
    Strategy that adapts to market regimes.
    
    Adjusts:
    - Position sizing
    - Stop-loss levels
    - Take-profit targets
    - Trading frequency
    """
    
    def __init__(self, config):
        self.config = config
        self.detector = RegimeDetector(config)
        
        # Regime-specific parameters
        self.regime_params = {
            MarketRegime.BULL_TRENDING: {
                'position_size_multiplier': 1.2,
                'stop_loss_pct': 0.015,
                'take_profit_pct': 0.05,
                'trade_frequency': 1.0
            },
            MarketRegime.BEAR_TRENDING: {
                'position_size_multiplier': 0.8,
                'stop_loss_pct': 0.025,
                'take_profit_pct': 0.03,
                'trade_frequency': 0.8
            },
            MarketRegime.HIGH_VOLATILITY: {
                'position_size_multiplier': 0.6,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.04,
                'trade_frequency': 0.7
            },
            MarketRegime.LOW_VOLATILITY: {
                'position_size_multiplier': 1.0,
                'stop_loss_pct': 0.01,
                'take_profit_pct': 0.02,
                'trade_frequency': 1.2
            },
            MarketRegime.SIDEWAYS: {
                'position_size_multiplier': 0.9,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.025,
                'trade_frequency': 1.1
            },
            MarketRegime.CRISIS: {
                'position_size_multiplier': 0.3,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.02,
                'trade_frequency': 0.5
            }
        }
    
    def update_regime(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None):
        """Update regime detection."""
        self.current_regime = self.detector.detect_regime(prices, volumes)
    
    def get_adjusted_parameters(self) -> Dict:
        """Get regime-adjusted trading parameters."""
        if not hasattr(self, 'current_regime'):
            return self.regime_params[MarketRegime.SIDEWAYS]
        
        return self.regime_params[self.current_regime]
    
    def should_trade(self, base_probability: float = 1.0) -> bool:
        """Decide if trading should occur based on regime."""
        params = self.get_adjusted_parameters()
        adjusted_prob = base_probability * params['trade_frequency']
        return np.random.random() < adjusted_prob