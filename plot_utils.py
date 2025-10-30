# plot_utils.py - PRODUCTION-READY PLOTTING
"""
Complete visualization utilities for:
- Equity curves
- Trading signals
- Performance metrics
- Training progress
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Directory for saving plots
PLOT_DIR = "runs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def plot_equity_curve(equity: np.ndarray, trades: Optional[List[dict]] = None,
                     title: str = "Equity Curve", save_path: Optional[str] = None) -> go.Figure:
    """
    Plot equity curve with optional trade markers.
    
    Args:
        equity: array of equity values
        trades: list of trade dicts with 'step', 'action', 'pnl'
        title: plot title
        save_path: path to save HTML
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Equity curve
    fig.add_trace(go.Scatter(
        y=equity,
        mode='lines',
        name='Equity',
        line=dict(color='blue', width=2)
    ))
    
    # Initial balance line
    if len(equity) > 0:
        fig.add_hline(y=equity[0], line_dash="dash", line_color="gray",
                     annotation_text="Initial Balance")
    
    # Trade markers
    if trades:
        buy_steps = [t['step'] for t in trades if t.get('action') == 'BUY']
        sell_steps = [t['step'] for t in trades if t.get('action') == 'SELL']
        
        if buy_steps:
            buy_equity = [equity[min(s, len(equity)-1)] for s in buy_steps]
            fig.add_trace(go.Scatter(
                x=buy_steps, y=buy_equity,
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
        
        if sell_steps:
            sell_equity = [equity[min(s, len(equity)-1)] for s in sell_steps]
            fig.add_trace(go.Scatter(
                x=sell_steps, y=sell_equity,
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Equity (USD)",
        template="plotly_dark",
        hovermode='x unified',
        width=1200,
        height=600
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"💾 Equity curve saved: {save_path}")
    
    return fig


def plot_training_metrics(metrics: Dict[str, List[float]], 
                          save_path: Optional[str] = None) -> go.Figure:
    """
    Plot training metrics over time.
    
    Args:
        metrics: dict with 'episode_rewards', 'episode_sharpes', etc.
        save_path: path to save HTML
    
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Sharpe Ratio', 'Win Rate', 'Losses'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    episodes = np.arange(len(metrics.get('episode_rewards', [])))
    
    # Episode rewards
    if 'episode_rewards' in metrics:
        fig.add_trace(go.Scatter(
            x=episodes, y=metrics['episode_rewards'],
            mode='lines+markers',
            name='Reward',
            line=dict(color='blue')
        ), row=1, col=1)
    
    # Sharpe ratio
    if 'episode_sharpes' in metrics:
        fig.add_trace(go.Scatter(
            x=episodes, y=metrics['episode_sharpes'],
            mode='lines+markers',
            name='Sharpe',
            line=dict(color='green')
        ), row=1, col=2)
    
    # Win rate
    if 'episode_win_rates' in metrics:
        fig.add_trace(go.Scatter(
            x=episodes, y=metrics['episode_win_rates'],
            mode='lines+markers',
            name='Win Rate',
            line=dict(color='orange')
        ), row=2, col=1)
    
    # Losses
    if 'losses' in metrics and 'total' in metrics['losses']:
        steps = np.arange(len(metrics['losses']['total']))
        fig.add_trace(go.Scatter(
            x=steps, y=metrics['losses']['total'],
            mode='lines',
            name='Total Loss',
            line=dict(color='red')
        ), row=2, col=2)
        
        if 'policy' in metrics['losses']:
            fig.add_trace(go.Scatter(
                x=steps, y=metrics['losses']['policy'],
                mode='lines',
                name='Policy Loss',
                line=dict(color='purple', dash='dash')
            ), row=2, col=2)
    
    fig.update_layout(
        template="plotly_dark",
        height=800,
        width=1400,
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"💾 Training metrics saved: {save_path}")
    
    return fig


def plot_candles_with_signals(df: pd.DataFrame, trades: Optional[List[dict]] = None,
                              symbol: str = "BTC/USDT", 
                              save_path: Optional[str] = None) -> go.Figure:
    """
    Plot candlestick chart with trade signals.
    
    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        trades: list of trade dicts
        symbol: market symbol
        save_path: path to save HTML
    
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='lime',
        decreasing_line_color='red'
    ), row=1, col=1)
    
    # Volume bars
    colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
              for i in range(len(df))]
    
    fig.add_trace(go.Bar(
        x=df.index if isinstance(df.index, pd.DatetimeIndex) else df['timestamp'],
        y=df['volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.5
    ), row=2, col=1)
    
    # Trade signals
    if trades:
        buy_trades = [t for t in trades if t.get('action') == 'BUY']
        sell_trades = [t for t in trades if t.get('action') == 'SELL']
        
        if buy_trades:
            buy_x = [df.index[t['step']] if isinstance(df.index, pd.DatetimeIndex) 
                    else df['timestamp'].iloc[t['step']] for t in buy_trades]
            buy_y = [t['price'] for t in buy_trades]
            
            fig.add_trace(go.Scatter(
                x=buy_x, y=buy_y,
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=15, symbol='triangle-up',
                           line=dict(color='white', width=2))
            ), row=1, col=1)
        
        if sell_trades:
            sell_x = [df.index[t['step']] if isinstance(df.index, pd.DatetimeIndex)
                     else df['timestamp'].iloc[t['step']] for t in sell_trades]
            sell_y = [t['price'] for t in sell_trades]
            
            fig.add_trace(go.Scatter(
                x=sell_x, y=sell_y,
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', size=15, symbol='triangle-down',
                           line=dict(color='white', width=2))
            ), row=1, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=800,
        width=1400,
        hovermode='x unified'
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"💾 Chart saved: {save_path}")
    
    return fig


def plot_drawdown(equity: np.ndarray, title: str = "Drawdown Analysis",
                 save_path: Optional[str] = None) -> go.Figure:
    """
    Plot drawdown over time.
    
    Args:
        equity: array of equity values
        title: plot title
        save_path: path to save HTML
    
    Returns:
        Plotly figure
    """
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Equity', 'Drawdown')
    )
    
    # Equity
    fig.add_trace(go.Scatter(
        y=equity,
        mode='lines',
        name='Equity',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        y=peak,
        mode='lines',
        name='Peak',
        line=dict(color='green', dash='dash')
    ), row=1, col=1)
    
    # Drawdown
    fig.add_trace(go.Scatter(
        y=-drawdown * 100,  # Convert to percentage
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='red')
    ), row=2, col=1)
    
    fig.update_yaxes(title_text="Equity (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=700,
        width=1200,
        hovermode='x unified'
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"💾 Drawdown plot saved: {save_path}")
    
    return fig


def plot_returns_distribution(returns: np.ndarray, 
                              title: str = "Returns Distribution",
                              save_path: Optional[str] = None) -> go.Figure:
    """
    Plot returns distribution histogram.
    
    Args:
        returns: array of returns
        title: plot title
        save_path: path to save HTML
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=returns * 100,  # Convert to percentage
        nbinsx=50,
        name='Returns',
        marker=dict(color='blue', line=dict(color='white', width=1))
    ))
    
    # Add mean line
    mean_return = returns.mean() * 100
    fig.add_vline(x=mean_return, line_dash="dash", line_color="green",
                 annotation_text=f"Mean: {mean_return:.2f}%")
    
    # Add zero line
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title=title,
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=500,
        width=800
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"💾 Returns distribution saved: {save_path}")
    
    return fig


def create_performance_dashboard(equity: np.ndarray, trades: List[dict],
                                 metrics: Dict, save_dir: str = PLOT_DIR):
    """
    Create comprehensive performance dashboard.
    
    Args:
        equity: equity curve
        trades: list of trades
        metrics: performance metrics
        save_dir: directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Equity curve
    plot_equity_curve(
        equity, trades,
        title="Equity Curve with Trade Signals",
        save_path=os.path.join(save_dir, "equity_curve.html")
    )
    
    # Drawdown
    plot_drawdown(
        equity,
        title="Drawdown Analysis",
        save_path=os.path.join(save_dir, "drawdown.html")
    )
    
    # Returns distribution
    if len(equity) > 1:
        returns = np.diff(equity) / equity[:-1]
        plot_returns_distribution(
            returns,
            title="Returns Distribution",
            save_path=os.path.join(save_dir, "returns_distribution.html")
        )
    
    logger.info(f"✅ Performance dashboard created in: {save_dir}")