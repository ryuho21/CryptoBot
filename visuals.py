import plotly.graph_objects as go
import numpy as np

def fib_levels(high, low):
    diff = high - low
    return [high - diff * r for r in [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]]

def plot_signals_with_indicators(df, signals=None, ghosts=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="price"))
    if "ema20" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema20"], name="EMA20"))
    if signals:
        for s in signals:
            color = "green" if s["direction"] == "LONG" else "red"
            fig.add_trace(go.Scatter(x=[s["time"]], y=[s["entry"]], mode="markers+text",
                                     marker=dict(color=color, size=10), text=[s["direction"]]))
    if ghosts is not None:
        fig.add_trace(go.Candlestick(x=ghosts["timestamp"], open=ghosts["open"], high=ghosts["high"], low=ghosts["low"], close=ghosts["close"], opacity=0.4, name="pred"))
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()
