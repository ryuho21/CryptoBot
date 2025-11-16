import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ========== TECHNICAL INDICATORS ==========

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, window=20, num_std=2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands."""
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def atr(df: pd.DataFrame, period=14) -> pd.Series:
    """Average True Range."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ========== PATTERN DETECTION ==========

def detect_fvg(df: pd.DataFrame) -> List[Dict]:
    """
    Detect Fair Value Gaps.
    
    Returns: List of FVG zones
    """
    zones = []
    
    for i in range(2, len(df)):
        # Bullish FVG
        gap = df['low'].iloc[i] - df['high'].iloc[i-2]
        if gap > 0:
            zones.append({
                'type': 'bullish',
                'idx': i,
                'top': df['low'].iloc[i],
                'bottom': df['high'].iloc[i-2],
                'timestamp': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i
            })
        
        # Bearish FVG
        gap = df['low'].iloc[i-2] - df['high'].iloc[i]
        if gap > 0:
            zones.append({
                'type': 'bearish',
                'idx': i,
                'top': df['low'].iloc[i-2],
                'bottom': df['high'].iloc[i],
                'timestamp': df['timestamp'].iloc[i] if 'timestamp' in df.columns else i
            })
    
    return zones


def detect_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
    """Detect support and resistance levels."""
    high_rolling_max = df['high'].rolling(window=window, center=True).max()
    low_rolling_min = df['low'].rolling(window=window, center=True).min()
    
    resistance = df[df['high'] == high_rolling_max]['high'].unique()
    support = df[df['low'] == low_rolling_min]['low'].unique()
    
    return {
        'support': sorted(support)[-5:],  # Last 5
        'resistance': sorted(resistance)[-5:]
    }


def fib_levels(high: float, low: float) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels."""
    diff = high - low
    return {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff,
        '1.0': low
    }


# ========== MAIN PLOTTING FUNCTION ==========

def plot_signals_with_indicators(df: pd.DataFrame, 
                                 signals: Optional[List[Dict]] = None, 
                                 ghosts: Optional[pd.DataFrame] = None,
                                 show_fvg: bool = True,
                                 show_fib: bool = True,
                                 show_sr: bool = True,
                                 show_volume: bool = True) -> go.Figure:
    """
    IMPROVED VERSION: Plot comprehensive trading chart.
    
    Args:
        df: OHLCV DataFrame with columns [timestamp, open, high, low, close, volume]
        signals: List of dicts with 'time', 'entry', 'direction' (or 'timestamp', 'price', 'side')
        ghosts: Ghost candles (predictions) DataFrame
        show_fvg: Show Fair Value Gaps
        show_fib: Show Fibonacci levels
        show_sr: Show Support/Resistance
        show_volume: Show volume subplot
    
    Returns:
        Plotly Figure
    """
    # Add technical indicators
    df = df.copy()
    df['ema20'] = ema(df['close'], 20)
    df['ema50'] = ema(df['close'], 50)
    df['rsi'] = rsi(df['close'])
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = bollinger_bands(df['close'])
    
    # Create subplots
    rows = 4 if show_volume else 3
    row_heights = [0.5, 0.15, 0.15, 0.2] if show_volume else [0.6, 0.2, 0.2]
    
    subplot_titles = ['Price & Signals', 'RSI', 'MACD']
    if show_volume:
        subplot_titles.append('Volume')
    
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles
    )
    
    # 1. CANDLESTICK
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'] if 'timestamp' in df.columns else df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # 2. EMAs
    if 'ema20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                y=df['ema20'],
                name='EMA 20',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )
    
    if 'ema50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                y=df['ema50'],
                name='EMA 50',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )
    
    # 3. BOLLINGER BANDS
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                y=df['bb_upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                y=df['bb_lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                opacity=0.5
            ),
            row=1, col=1
        )
    
    # 4. TRADE SIGNALS
    if signals:
        for sig in signals:
            # Support both formats
            timestamp = sig.get('time') or sig.get('timestamp')
            price = sig.get('entry') or sig.get('price')
            direction = sig.get('direction') or sig.get('side', 'UNKNOWN')
            
            # Determine color
            if direction.upper() in ['LONG', 'BUY']:
                color = '#00ff00'
                symbol = 'triangle-up'
                text_pos = 'bottom center'
            else:
                color = '#ff0000'
                symbol = 'triangle-down'
                text_pos = 'top center'
            
            fig.add_trace(
                go.Scatter(
                    x=[timestamp],
                    y=[price],
                    mode='markers+text',
                    name=f'{direction} Signal',
                    marker=dict(color=color, size=15, symbol=symbol, line=dict(color='white', width=2)),
                    text=[direction.upper()],
                    textposition=text_pos,
                    textfont=dict(size=10, color='white'),
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # 5. GHOST CANDLES (Predictions)
    if ghosts is not None and len(ghosts) > 0:
        fig.add_trace(
            go.Candlestick(
                x=ghosts['timestamp'] if 'timestamp' in ghosts.columns else ghosts.index,
                open=ghosts['open'],
                high=ghosts['high'],
                low=ghosts['low'],
                close=ghosts['close'],
                name='Prediction',
                increasing_line_color='cyan',
                decreasing_line_color='orange',
                opacity=0.4
            ),
            row=1, col=1
        )
    
    # 6. FAIR VALUE GAPS
    if show_fvg:
        fvg_zones = detect_fvg(df)
        for zone in fvg_zones[-20:]:  # Last 20
            color = 'rgba(0, 255, 0, 0.1)' if zone['type'] == 'bullish' else 'rgba(255, 0, 0, 0.1)'
            
            fig.add_shape(
                type="rect",
                x0=zone['timestamp'],
                x1=df['timestamp'].iloc[-1] if 'timestamp' in df.columns else df.index[-1],
                y0=zone['bottom'],
                y1=zone['top'],
                fillcolor=color,
                line_width=0,
                layer="below",
                row=1, col=1
            )
    
    # 7. SUPPORT/RESISTANCE
    if show_sr and len(df) > 20:
        sr_levels = detect_support_resistance(df)
        
        for level in sr_levels.get('support', []):
            fig.add_hline(
                y=level,
                line_dash="dot",
                line_color="green",
                opacity=0.5,
                annotation_text=f"S: {level:.2f}",
                annotation_position="right",
                row=1, col=1
            )
        
        for level in sr_levels.get('resistance', []):
            fig.add_hline(
                y=level,
                line_dash="dot",
                line_color="red",
                opacity=0.5,
                annotation_text=f"R: {level:.2f}",
                annotation_position="right",
                row=1, col=1
            )
    
    # 8. FIBONACCI LEVELS
    if show_fib and len(df) > 50:
        lookback = min(200, len(df))
        high = df['high'].tail(lookback).max()
        low = df['low'].tail(lookback).min()
        fib_lvls = fib_levels(high, low)
        
        for level_name, level_value in fib_lvls.items():
            if 0 <= float(level_name) <= 1:
                fig.add_hline(
                    y=level_value,
                    line_dash="dash",
                    line_color="yellow",
                    opacity=0.3,
                    annotation_text=f"Fib {level_name}",
                    annotation_position="left",
                    row=1, col=1
                )
    
    # 9. RSI SUBPLOT
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                y=df['rsi'],
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
    
    # 10. MACD SUBPLOT
    if 'macd' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                y=df['macd'],
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        
        if 'macd_signal' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                    y=df['macd_signal'],
                    name='Signal',
                    line=dict(color='orange', width=2)
                ),
                row=3, col=1
            )
        
        if 'macd_hist' in df.columns:
            colors = ['#26a69a' if val >= 0 else '#ef5350' for val in df['macd_hist']]
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                    y=df['macd_hist'],
                    name='Histogram',
                    marker_color=colors
                ),
                row=3, col=1
            )
    
    # 11. VOLUME SUBPLOT
    if show_volume and 'volume' in df.columns:
        colors = ['rgba(38, 166, 154, 0.5)' if df['close'].iloc[i] >= df['open'].iloc[i] 
                 else 'rgba(239, 83, 80, 0.5)' for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors
            ),
            row=4, col=1
        )
        
        if len(df) > 20:
            vol_sma = df['volume'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                    y=vol_sma,
                    name='Vol SMA',
                    line=dict(color='yellow', width=1.5)
                ),
                row=4, col=1
            )
    
    # LAYOUT
    fig.update_layout(
        height=1000,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
    
    return fig


# ========== BACKWARD COMPATIBILITY ==========
# Keep your original function signature working
def plot_signals(df, signals=None, ghosts=None):
    """Alias for backward compatibility."""
    return plot_signals_with_indicators(df, signals, ghosts)


# ========== USAGE EXAMPLE ==========
if __name__ == '__main__':
    print("=" * 70)
    print("📊 Enhanced Visuals - Demo")
    print("=" * 70)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    base_price = 40000
    trend = np.linspace(0, 1000, 1000)
    noise = np.random.normal(0, 100, 1000)
    prices = base_price + trend + noise
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(noise) * 0.5,
        'low': prices - np.abs(noise) * 0.5,
        'close': prices + np.random.normal(0, 20, 1000),
        'volume': np.random.uniform(10, 100, 1000)
    })
    
    # Create sample signals
    signals = [
        {'time': dates[100], 'entry': prices[100], 'direction': 'LONG'},
        {'time': dates[300], 'entry': prices[300], 'direction': 'SHORT'},
        {'time': dates[500], 'entry': prices[500], 'direction': 'LONG'},
    ]
    
    # Generate chart
    fig = plot_signals_with_indicators(
        df,
        signals=signals,
        show_fvg=True,
        show_fib=True,
        show_sr=True,
        show_volume=True
    )
    
    print("\n✅ Chart generated with:")
    print("   - Candlesticks")
    print("   - EMAs (20, 50)")
    print("   - Bollinger Bands")
    print("   - RSI")
    print("   - MACD")
    print("   - Volume")
    print("   - Trade signals")
    print("   - Fair Value Gaps")
    print("   - Fibonacci levels")
    print("   - Support/Resistance")
    
    # Save
    import os
    os.makedirs("runs/plots", exist_ok=True)
    fig.write_html("runs/plots/enhanced_chart.html")
    print(f"\n💾 Saved to: runs/plots/enhanced_chart.html")
    print("=" * 70)
