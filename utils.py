# utils.py - PRODUCTION-READY UTILITIES
"""
Comprehensive utility functions for:
- File management
- Checkpointing
- Metrics calculation
- Data processing
"""

import os
import json
import glob
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd


def make_runs_dir(path: str):
    """Create runs directory structure."""
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(path, "plots"), exist_ok=True)
    os.makedirs(os.path.join(path, "logs"), exist_ok=True)


def save_jsonl(path: str, obj: dict):
    """Append object to JSONL file."""
    obj["timestamp"] = datetime.now().isoformat()
    
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def load_jsonl(path: str) -> List[dict]:
    """Load all objects from JSONL file."""
    if not os.path.exists(path):
        return []
    
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    return data


def latest_checkpoint(pattern: str = "runs/checkpoints/checkpoint*.pt") -> Optional[str]:
    """Find most recent checkpoint file."""
    candidates = glob.glob(pattern)
    
    if not candidates:
        return None
    
    return max(candidates, key=os.path.getmtime)


def save_checkpoint(path: str, state: Dict):
    """Save checkpoint with metadata."""
    state["saved_at"] = datetime.now().isoformat()
    state["saved_timestamp"] = time.time()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device) -> Dict:
    """Load checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: array of returns
        risk_free_rate: risk-free rate (annual)
        periods: trading periods per year (252 for daily, 252*24*60 for minute)
    
    Returns:
        Sharpe ratio
    """
    r = np.array(returns)
    
    if r.size == 0 or r.std() == 0:
        return 0.0
    
    excess_returns = r - risk_free_rate / periods
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods)
    
    return float(sharpe)


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, periods: int = 252) -> float:
    """
    Calculate annualized Sortino ratio.
    
    Args:
        returns: array of returns
        risk_free_rate: risk-free rate (annual)
        periods: trading periods per year
    
    Returns:
        Sortino ratio
    """
    r = np.array(returns)
    
    if r.size == 0:
        return 0.0
    
    excess_returns = r - risk_free_rate / periods
    downside_returns = excess_returns[excess_returns < 0]
    
    if downside_returns.size == 0 or downside_returns.std() == 0:
        return 0.0
    
    sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(periods)
    
    return float(sortino)


def max_drawdown(equity: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        equity: array of equity values
    
    Returns:
        Maximum drawdown (0 to 1)
    """
    eq = np.array(equity)
    
    if eq.size == 0:
        return 0.0
    
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / (peak + 1e-9)
    
    return float(np.max(drawdown))


def calmar_ratio(returns: np.ndarray, equity: np.ndarray, periods: int = 252) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Args:
        returns: array of returns
        equity: array of equity values
        periods: trading periods per year
    
    Returns:
        Calmar ratio
    """
    r = np.array(returns)
    
    if r.size == 0:
        return 0.0
    
    annual_return = r.mean() * periods
    max_dd = max_drawdown(equity)
    
    if max_dd == 0:
        return 0.0
    
    return float(annual_return / max_dd)


def calculate_metrics(equity_curve: np.ndarray, trades: List[dict]) -> Dict[str, float]:
    """
    Calculate comprehensive trading metrics.
    
    Args:
        equity_curve: array of equity values over time
        trades: list of trade dictionaries
    
    Returns:
        Dictionary of metrics
    """
    equity = np.array(equity_curve)
    
    if len(equity) < 2:
        return {}
    
    # Returns
    returns = np.diff(equity) / equity[:-1]
    
    # Basic metrics
    total_return = (equity[-1] - equity[0]) / equity[0]
    
    # Risk metrics
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    max_dd = max_drawdown(equity)
    calmar = calmar_ratio(returns, equity)
    
    # Trade metrics
    winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
    losing_trades = sum(1 for t in trades if t.get('pnl', 0) < 0)
    total_trades = winning_trades + losing_trades
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # Profit factor
    gross_profit = sum(t['pnl'] for t in trades if t.get('pnl', 0) > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t.get('pnl', 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    
    # Average metrics
    avg_win = gross_profit / winning_trades if winning_trades > 0 else 0.0
    avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0.0
    
    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'max_drawdown': float(max_dd),
        'calmar_ratio': float(calmar),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'final_equity': float(equity[-1])
    }


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)


def exponential_moving_average(data: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Calculate exponential moving average.
    
    Args:
        data: input array
        alpha: smoothing factor (0 to 1)
    
    Returns:
        EMA array
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]
    
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    
    return ema


def normalize_array(arr: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize array.
    
    Args:
        arr: input array
        method: 'zscore', 'minmax', or 'robust'
    
    Returns:
        Normalized array
    """
    if method == 'zscore':
        mean = arr.mean()
        std = arr.std()
        return (arr - mean) / (std + 1e-8)
    
    elif method == 'minmax':
        min_val = arr.min()
        max_val = arr.max()
        return (arr - min_val) / (max_val - min_val + 1e-8)
    
    elif method == 'robust':
        median = np.median(arr)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25
        return (arr - median) / (iqr + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_summary_report(metrics: Dict, config: Dict, save_path: str):
    """
    Create HTML summary report.
    
    Args:
        metrics: performance metrics
        config: configuration dict
        save_path: path to save report
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Bot Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; color: #27ae60; }}
            .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>🤖 Trading Bot Performance Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>📊 Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Return</td><td class="metric">{metrics.get('total_return', 0):.2%}</td></tr>
            <tr><td>Sharpe Ratio</td><td class="metric">{metrics.get('sharpe_ratio', 0):.2f}</td></tr>
            <tr><td>Sortino Ratio</td><td class="metric">{metrics.get('sortino_ratio', 0):.2f}</td></tr>
            <tr><td>Max Drawdown</td><td class="metric">{metrics.get('max_drawdown', 0):.2%}</td></tr>
            <tr><td>Calmar Ratio</td><td class="metric">{metrics.get('calmar_ratio', 0):.2f}</td></tr>
            <tr><td>Win Rate</td><td class="metric">{metrics.get('win_rate', 0):.2%}</td></tr>
            <tr><td>Profit Factor</td><td class="metric">{metrics.get('profit_factor', 0):.2f}</td></tr>
        </table>
        
        <h2>📈 Trading Statistics</h2>
        <table>
            <tr><th>Statistic</th><th>Value</th></tr>
            <tr><td>Total Trades</td><td>{metrics.get('total_trades', 0)}</td></tr>
            <tr><td>Winning Trades</td><td>{metrics.get('winning_trades', 0)}</td></tr>
            <tr><td>Losing Trades</td><td>{metrics.get('losing_trades', 0)}</td></tr>
            <tr><td>Average Win</td><td>${metrics.get('avg_win', 0):.2f}</td></tr>
            <tr><td>Average Loss</td><td>${metrics.get('avg_loss', 0):.2f}</td></tr>
            <tr><td>Final Equity</td><td>${metrics.get('final_equity', 0):,.2f}</td></tr>
        </table>
        
        <h2>⚙️ Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Symbol</td><td>{config.get('symbol', 'N/A')}</td></tr>
            <tr><td>Timeframe</td><td>{config.get('timeframe', 'N/A')}</td></tr>
            <tr><td>Initial Balance</td><td>${config.get('initial_balance', 0):,.2f}</td></tr>
            <tr><td>Learning Rate (Actor)</td><td>{config.get('learning_rate_actor', 0):.2e}</td></tr>
            <tr><td>Learning Rate (Critic)</td><td>{config.get('learning_rate_critic', 0):.2e}</td></tr>
            <tr><td>Gamma</td><td>{config.get('gamma', 0):.3f}</td></tr>
            <tr><td>GAE Lambda</td><td>{config.get('gae_lambda', 0):.3f}</td></tr>
        </table>
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(html)
    
    print(f"✅ Summary report saved: {save_path}")