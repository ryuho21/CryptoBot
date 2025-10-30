# multi_asset_manager.py - PRODUCTION-READY MULTI-ASSET SUPPORT
"""
Complete multi-asset trading system with:
- Portfolio management
- Cross-asset correlation
- Dynamic rebalancing
- Risk allocation
- Graph neural networks
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from config import Config
from graph_encoder import AssetGraphEncoder

logger = logging.getLogger(__name__)


@dataclass
class AssetAllocation:
    """Asset allocation state."""
    symbol: str
    weight: float
    position: float
    value: float
    pnl: float


class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory.
    
    Features:
    - Mean-variance optimization
    - Risk parity
    - Maximum Sharpe ratio
    - Minimum volatility
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.risk_free_rate = 0.02  # 2% annual
    
    def optimize_weights(self, returns: np.ndarray, 
                        method: str = 'max_sharpe') -> np.ndarray:
        """
        Optimize portfolio weights.
        
        Args:
            returns: (T, N) array of asset returns
            method: 'max_sharpe', 'min_vol', 'risk_parity', 'equal'
        
        Returns:
            weights: (N,) optimal weights
        """
        n_assets = returns.shape[1]
        
        if method == 'equal':
            return np.ones(n_assets) / n_assets
        
        # Calculate statistics
        mean_returns = returns.mean(axis=0)
        cov_matrix = np.cov(returns.T)
        
        if method == 'max_sharpe':
            return self._max_sharpe_weights(mean_returns, cov_matrix)
        elif method == 'min_vol':
            return self._min_volatility_weights(cov_matrix)
        elif method == 'risk_parity':
            return self._risk_parity_weights(cov_matrix)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _max_sharpe_weights(self, mean_returns: np.ndarray, 
                           cov_matrix: np.ndarray) -> np.ndarray:
        """Maximum Sharpe ratio weights."""
        from scipy.optimize import minimize
        
        n_assets = len(mean_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate / 252) / portfolio_vol
            return -sharpe
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(negative_sharpe, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_weights
    
    def _min_volatility_weights(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Minimum volatility weights."""
        from scipy.optimize import minimize
        
        n_assets = len(cov_matrix)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(portfolio_volatility, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_weights
    
    def _risk_parity_weights(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Risk parity weights."""
        from scipy.optimize import minimize
        
        n_assets = len(cov_matrix)
        
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            
            # Minimize difference from equal risk contribution
            target = portfolio_vol / n_assets
            return np.sum((risk_contrib - target) ** 2)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(risk_parity_objective, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_weights


class MultiAssetAgent:
    """
    Multi-asset trading agent with graph neural networks.
    
    Features:
    - Cross-asset feature learning
    - Correlation-aware decisions
    - Dynamic portfolio rebalancing
    """
    
    def __init__(self, config: Config, symbols: List[str]):
        self.config = config
        self.symbols = symbols
        self.num_assets = len(symbols)
        self.device = config.device
        
        # Graph encoder for cross-asset learning
        self.graph_encoder = AssetGraphEncoder(
            input_dim=config.feature_dim,
            hidden_dim=config.hidden_size,
            output_dim=config.hidden_size,
            num_assets=self.num_assets,
            use_gat=True
        ).to(self.device)
        
        # Portfolio optimizer
        self.optimizer = PortfolioOptimizer(config)
        
        # Asset allocations
        self.allocations: Dict[str, AssetAllocation] = {}
        for symbol in symbols:
            self.allocations[symbol] = AssetAllocation(
                symbol=symbol,
                weight=1.0 / self.num_assets,
                position=0.0,
                value=0.0,
                pnl=0.0
            )
    
    def process_multi_asset_features(self, features: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Process features from multiple assets using graph encoder.
        
        Args:
            features: dict of {symbol: (W, F) features}
        
        Returns:
            encoded_features: (num_assets, H) encoded features
        """
        # Stack features from all assets
        asset_features = []
        for symbol in self.symbols:
            if symbol in features:
                # Take last timestep features
                last_features = features[symbol][-1, :]
                asset_features.append(last_features)
            else:
                # Missing asset - use zeros
                asset_features.append(np.zeros(self.config.feature_dim))
        
        asset_features = np.array(asset_features)  # (N, F)
        asset_features_t = torch.from_numpy(asset_features).float().to(self.device)
        
        # Encode with graph network
        with torch.no_grad():
            encoded = self.graph_encoder(asset_features_t)
        
        return encoded
    
    def calculate_correlation_matrix(self, returns_history: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate correlation matrix across assets.
        
        Args:
            returns_history: dict of {symbol: returns array}
        
        Returns:
            correlation_matrix: (N, N)
        """
        returns_list = []
        for symbol in self.symbols:
            if symbol in returns_history:
                returns_list.append(returns_history[symbol])
            else:
                returns_list.append(np.zeros(100))  # Placeholder
        
        returns_matrix = np.column_stack(returns_list)
        correlation = np.corrcoef(returns_matrix.T)
        
        return correlation
    
    def rebalance_portfolio(self, current_allocations: Dict[str, float],
                           target_weights: np.ndarray,
                           total_capital: float) -> Dict[str, float]:
        """
        Calculate rebalancing trades.
        
        Args:
            current_allocations: dict of {symbol: current_value}
            target_weights: (N,) target weights
            total_capital: total portfolio value
        
        Returns:
            trades: dict of {symbol: trade_amount (+ buy, - sell)}
        """
        trades = {}
        
        for i, symbol in enumerate(self.symbols):
            target_value = total_capital * target_weights[i]
            current_value = current_allocations.get(symbol, 0.0)
            trade_amount = target_value - current_value
            
            # Minimum trade threshold
            if abs(trade_amount) > 100:
                trades[symbol] = trade_amount
        
        return trades
    
    def update_allocations(self, symbol: str, position: float, 
                          value: float, pnl: float):
        """Update allocation for specific asset."""
        if symbol in self.allocations:
            self.allocations[symbol].position = position
            self.allocations[symbol].value = value
            self.allocations[symbol].pnl = pnl
    
    def get_portfolio_statistics(self) -> Dict:
        """Calculate portfolio-level statistics."""
        total_value = sum(a.value for a in self.allocations.values())
        total_pnl = sum(a.pnl for a in self.allocations.values())
        
        # Actual weights
        actual_weights = {
            symbol: alloc.value / total_value if total_value > 0 else 0
            for symbol, alloc in self.allocations.items()
        }
        
        return {
            'total_value': total_value,
            'total_pnl': total_pnl,
            'allocations': {s: asdict(a) for s, a in self.allocations.items()},
            'actual_weights': actual_weights
        }


class RebalancingScheduler:
    """
    Portfolio rebalancing scheduler.
    
    Triggers rebalancing based on:
    - Time intervals
    - Drift thresholds
    - Market conditions
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.last_rebalance_step = 0
        self.rebalance_interval = getattr(config, 'rebalance_interval', 1000)
        self.drift_threshold = getattr(config, 'rebalance_drift_threshold', 0.05)
    
    def should_rebalance(self, current_step: int,
                        target_weights: np.ndarray,
                        actual_weights: np.ndarray) -> bool:
        """
        Check if rebalancing is needed.
        
        Args:
            current_step: current timestep
            target_weights: target allocation
            actual_weights: current allocation
        
        Returns:
            should_rebalance: bool
        """
        # Time-based trigger
        if current_step - self.last_rebalance_step >= self.rebalance_interval:
            logger.info(f"⏰ Time-based rebalancing triggered at step {current_step}")
            return True
        
        # Drift-based trigger
        max_drift = np.max(np.abs(target_weights - actual_weights))
        if max_drift > self.drift_threshold:
            logger.info(f"📊 Drift-based rebalancing triggered: max_drift={max_drift:.2%}")
            return True
        
        return False
    
    def mark_rebalanced(self, step: int):
        """Mark that rebalancing occurred."""
        self.last_rebalance_step = step