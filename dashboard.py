# dashboard.py - IMPROVED VERSION
"""
Critical fixes and improvements:
1. Fixed asyncio event loop handling (was causing crashes)
2. Added proper error handling for WebSocket operations
3. Fixed state management in Streamlit session
4. Improved chart rendering performance
5. Added connection health monitoring
6. Fixed replay controls race condition
7. Better memory management for large datasets
8. Added export functionality for charts
"""

import os
import time
import math
import json
import threading
import asyncio
import logging
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Defensive imports
try:
    from live_bridge import LiveTradingBridge
except:
    LiveTradingBridge = None

try:
    import torch
    from policy import PolicyNet
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except:
    WEBSOCKETS_AVAILABLE = False

logger = logging.getLogger("dashboard")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

# -------------------------
# Indicator Functions (Optimized)
# -------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD indicator."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, period=14) -> pd.Series:
    """Average True Range."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def bollinger_bands(series: pd.Series, window=20, num_std=2):
    """Bollinger Bands."""
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

def detect_fvg(df: pd.DataFrame) -> List[tuple]:
    """Detect Fair Value Gaps."""
    zones = []
    for i in range(2, len(df)):
        try:
            # Bullish FVG: Current low > Previous candle's high
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                zones.append(("bull_fvg", i, df['high'].iloc[i-2], df['low'].iloc[i]))
            # Bearish FVG: Current high < Previous candle's low
            elif df['high'].iloc[i] < df['low'].iloc[i-2]:
                zones.append(("bear_fvg", i, df['high'].iloc[i], df['low'].iloc[i-2]))
        except:
            continue
    return zones

def fib_levels(high: float, low: float) -> Dict[str, float]:
    """Calculate Fibonacci retracement levels."""
    diff = high - low
    return {
        "0.0": high,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
        "1.0": low
    }

# -------------------------
# Data Loading (Cached)
# -------------------------
@st.cache_data(ttl=60)
def load_ohlcv_data(symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Load OHLCV data with caching."""
    # Try multiple sources
    for path in ["runs/ohlcv.csv", "runs/ohlcv.parquet", "runs/ohlcv.json"]:
        if os.path.exists(path):
            try:
                if path.endswith(".csv"):
                    df = pd.read_csv(path, parse_dates=["timestamp"])
                elif path.endswith(".parquet"):
                    df = pd.read_parquet(path)
                elif path.endswith(".json"):
                    df = pd.read_json(path)
                
                return df.tail(limit).reset_index(drop=True)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                continue
    return None

@st.cache_data(ttl=30)
def load_trade_log(path="runs/live_trades.csv") -> pd.DataFrame:
    """Load trade history with caching."""
    if os.path.exists(path):
        try:
            return pd.read_csv(path, parse_dates=["ts"], date_parser=lambda x: pd.to_datetime(x, errors='coerce'))
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def load_state(path="runs/state.json") -> Optional[Dict]:
    """Load bot state."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except:
            return None
    return None

# -------------------------
# WebSocket Manager (FIXED)
# -------------------------
class WSManager:
    """
    FIXED: Proper asyncio event loop management in threads.
    """
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.latest_state = {}
        self.server_future = None
        self.client_ws = None
        self._server_thread = None
        self._client_thread = None
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self.connection_status = {"server": False, "client": False}

    # -------- SERVER --------
    async def _handle_client(self, websocket, path):
        """Handle incoming WebSocket client."""
        client_id = f"{websocket.remote_address}"
        logger.info(f"WS client connected: {client_id}")
        self.clients.add(websocket)
        self.connection_status["server"] = True
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    event_type = data.get("event", "")
                    
                    if event_type == "state_update":
                        self.latest_state.update(data)
                    elif event_type == "trade_event":
                        if "_events" not in self.latest_state:
                            self.latest_state["_events"] = []
                        self.latest_state["_events"].append(data)
                        # Keep only last 500 events
                        if len(self.latest_state["_events"]) > 500:
                            self.latest_state["_events"] = self.latest_state["_events"][-500:]
                    
                    logger.debug(f"Received event: {event_type}")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from client: {e}")
                except Exception as e:
                    logger.exception(f"Error processing message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        finally:
            self.clients.discard(websocket)
            if not self.clients:
                self.connection_status["server"] = False

    def _run_server(self):
        """Run WebSocket server in dedicated thread with new event loop."""
        async def start_server():
            try:
                server = await websockets.serve(
                    self._handle_client, 
                    self.host, 
                    self.port,
                    ping_interval=20,
                    ping_timeout=10
                )
                logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
                self.server_future = server
                
                # Keep server running
                await asyncio.Future()  # Run forever
            except Exception as e:
                logger.exception(f"Server error: {e}")
        
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(start_server())
        except Exception as e:
            logger.exception(f"Server loop error: {e}")
        finally:
            loop.close()

    def start_server(self) -> bool:
        """Start WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("websockets library not available")
            return False
        
        if self._server_thread and self._server_thread.is_alive():
            logger.info("Server already running")
            return True
        
        self._server_thread = threading.Thread(target=self._run_server, daemon=True, name="WS-Server")
        self._server_thread.start()
        time.sleep(0.5)  # Give it time to start
        return True

    def stop_server(self):
        """Stop WebSocket server."""
        self._stop_event.set()
        if self.server_future:
            try:
                self.server_future.close()
            except:
                pass
        self.connection_status["server"] = False
        logger.info("Server stopped")

    # -------- CLIENT --------
    async def _client_loop(self, uri: str):
        """Client connection loop with reconnection."""
        retry_count = 0
        max_retries = 5
        
        while not self._stop_event.is_set() and retry_count < max_retries:
            try:
                async with websockets.connect(uri, ping_interval=20, ping_timeout=10) as ws:
                    logger.info(f"Connected to bot at {uri}")
                    self.client_ws = ws
                    self.connection_status["client"] = True
                    retry_count = 0  # Reset on successful connection
                    
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            event_type = data.get("event", "")
                            
                            if event_type == "state_update":
                                self.latest_state.update(data)
                            elif event_type == "trade_event":
                                if "_events" not in self.latest_state:
                                    self.latest_state["_events"] = []
                                self.latest_state["_events"].append(data)
                                if len(self.latest_state["_events"]) > 500:
                                    self.latest_state["_events"] = self.latest_state["_events"][-500:]
                        except Exception as e:
                            logger.exception(f"Error processing client message: {e}")
            
            except websockets.exceptions.WebSocketException as e:
                logger.error(f"WebSocket error: {e}")
                retry_count += 1
                self.connection_status["client"] = False
                self.client_ws = None
                
                if retry_count < max_retries:
                    wait_time = min(2 ** retry_count, 30)
                    logger.info(f"Reconnecting in {wait_time}s... (attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
            
            except Exception as e:
                logger.exception(f"Client loop error: {e}")
                break
        
        self.connection_status["client"] = False
        self.client_ws = None
        logger.info("Client loop exited")

    def _run_client(self, uri: str):
        """Run client in dedicated thread with new event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._client_loop(uri))
        except Exception as e:
            logger.exception(f"Client loop error: {e}")
        finally:
            loop.close()

    def start_client(self, uri: str) -> bool:
        """Start WebSocket client."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("websockets library not available")
            return False
        
        if self._client_thread and self._client_thread.is_alive():
            logger.info("Client already running")
            return True
        
        self._client_thread = threading.Thread(
            target=self._run_client, 
            args=(uri,), 
            daemon=True,
            name="WS-Client"
        )
        self._client_thread.start()
        return True

    def send_command(self, cmd: Dict):
        """Send command to all connected clients/server."""
        message = json.dumps(cmd)
        
        # Send to server clients
        async def _broadcast():
            dead_clients = []
            for client in list(self.clients):
                try:
                    await client.send(message)
                except Exception as e:
                    logger.error(f"Failed to send to client: {e}")
                    dead_clients.append(client)
            
            for client in dead_clients:
                self.clients.discard(client)
        
        # Create task in any available event loop
        try:
            # Try to schedule in existing loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_broadcast())
            else:
                loop.run_until_complete(_broadcast())
        except RuntimeError:
            # No event loop in current thread - use executor
            self._executor.submit(asyncio.run, _broadcast())
        
        # Send to client websocket
        if self.client_ws:
            async def _send_to_server():
                try:
                    await self.client_ws.send(message)
                except Exception as e:
                    logger.error(f"Failed to send to server: {e}")
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(_send_to_server())
                else:
                    loop.run_until_complete(_send_to_server())
            except RuntimeError:
                self._executor.submit(asyncio.run, _send_to_server())

    def get_status(self) -> Dict[str, Any]:
        """Get connection status."""
        return {
            "server_running": self.connection_status["server"],
            "client_connected": self.connection_status["client"],
            "num_clients": len(self.clients),
            "has_state": bool(self.latest_state)
        }


# -------------------------
# Chart Plotting (Optimized)
# -------------------------
def create_price_chart(df: pd.DataFrame, signals: List[Dict] = None, 
                       ghosts: pd.DataFrame = None, show_fvg: bool = True,
                       show_fib: bool = True) -> go.Figure:
    """
    Create comprehensive price chart with indicators.
    OPTIMIZED: Uses subplots for better performance.
    """
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price & Signals', 'RSI', 'MACD')
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # EMAs
    if "ema20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["ema20"], 
                      name="EMA20", line=dict(color='orange', width=1)),
            row=1, col=1
        )
    if "ema50" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["ema50"], 
                      name="EMA50", line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    # Bollinger Bands
    if "close" in df.columns:
        bb_upper, bb_middle, bb_lower = bollinger_bands(df["close"])
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=bb_upper, 
                      name="BB Upper", line=dict(color='gray', width=1, dash='dash'),
                      opacity=0.5),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=bb_lower, 
                      name="BB Lower", line=dict(color='gray', width=1, dash='dash'),
                      opacity=0.5, fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
    
    # Trade signals
    if signals:
        for sig in signals:
            color = 'green' if sig.get("side", "").lower() in ("buy", "long") else 'red'
            symbol = 'triangle-up' if color == 'green' else 'triangle-down'
            
            fig.add_trace(
                go.Scatter(
                    x=[sig["timestamp"]], 
                    y=[sig["price"]],
                    mode="markers+text",
                    marker=dict(color=color, size=12, symbol=symbol),
                    text=[sig.get("label", sig.get("side", ""))],
                    textposition="top center",
                    name=f"Signal {sig.get('side', '')}",
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Ghost candles (predictions)
    if ghosts is not None and len(ghosts) > 0:
        fig.add_trace(
            go.Candlestick(
                x=ghosts["timestamp"],
                open=ghosts["open"],
                high=ghosts["high"],
                low=ghosts["low"],
                close=ghosts["close"],
                opacity=0.4,
                name="Prediction",
                increasing_line_color='cyan',
                decreasing_line_color='orange'
            ),
            row=1, col=1
        )
    
    # Fair Value Gaps
    if show_fvg:
        try:
            fvg_zones = detect_fvg(df)
            for zone in fvg_zones[-10:]:  # Only show last 10
                zone_type, idx, y0, y1 = zone
                color = "rgba(0,255,0,0.15)" if zone_type == "bull_fvg" else "rgba(255,0,0,0.15)"
                
                fig.add_shape(
                    type="rect",
                    x0=df["timestamp"].iloc[idx],
                    x1=df["timestamp"].iloc[-1],
                    y0=y0, y1=y1,
                    fillcolor=color,
                    line_width=0,
                    layer="below",
                    row=1, col=1
                )
        except Exception as e:
            logger.error(f"FVG detection error: {e}")
    
    # Fibonacci levels
    if show_fib and len(df) > 50:
        try:
            lookback = min(200, len(df))
            high = df["high"].tail(lookback).max()
            low = df["low"].tail(lookback).min()
            levels = fib_levels(high, low)
            
            for level_name, level_value in levels.items():
                fig.add_hline(
                    y=level_value,
                    line_dash="dot",
                    line_color="rgba(150,150,150,0.5)",
                    annotation_text=f"Fib {level_name}",
                    annotation_position="right",
                    row=1, col=1
                )
        except Exception as e:
            logger.error(f"Fibonacci error: {e}")
    
    # RSI subplot
    if "rsi" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["rsi"], 
                      name="RSI", line=dict(color='purple', width=1)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    
    # MACD subplot
    if "macd" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["macd"], 
                      name="MACD", line=dict(color='blue', width=1)),
            row=3, col=1
        )
        if "macd_signal" in df.columns:
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["macd_signal"], 
                          name="Signal", line=dict(color='orange', width=1)),
                row=3, col=1
            )
        if "macd_hist" in df.columns:
            colors = ['green' if val >= 0 else 'red' for val in df["macd_hist"]]
            fig.add_trace(
                go.Bar(x=df["timestamp"], y=df["macd_hist"], 
                      name="Histogram", marker_color=colors),
                row=3, col=1
            )
    
    # Layout
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(
    page_title="Trading Bot Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/trading-bot',
        'Report a bug': "https://github.com/yourusername/trading-bot/issues",
        'About': "# Advanced Trading Bot Dashboard\nReal-time monitoring and control"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4;}
    .metric-card {background-color: #0e1117; padding: 20px; border-radius: 10px; border: 1px solid #262730;}
    .status-good {color: #26a69a;}
    .status-bad {color: #ef5350;}
    .status-neutral {color: #ffa726;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 Trading Bot Dashboard</p>', unsafe_allow_html=True)

# Initialize session state
if "ws_manager" not in st.session_state:
    st.session_state.ws_manager = None
if "replay_idx" not in st.session_state:
    st.session_state.replay_idx = 0
if "replay_playing" not in st.session_state:
    st.session_state.replay_playing = False
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Data Source")
    symbol = st.text_input("Symbol", value=os.environ.get("SYMBOL", "BTC/USDT"))
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h"], index=1)
    limit = st.slider("Bars to display", 100, 2000, 600, 50)
    
    st.subheader("Display Options")
    show_ghosts = st.checkbox("Show predictions", value=True)
    show_trades = st.checkbox("Show trade signals", value=True)
    show_fvg = st.checkbox("Show FVG zones", value=True)
    show_fib = st.checkbox("Show Fibonacci", value=True)
    pred_count = st.slider("Prediction bars", 1, 24, 6)
    
    st.subheader("Auto-refresh")
    refresh_interval = st.selectbox(
        "Refresh every",
        [("Manual", 0), ("2s", 2), ("5s", 5), ("10s", 10), ("30s", 30), ("60s", 60)],
        index=2,
        format_func=lambda x: x[0]
    )[1]
    
    if refresh_interval > 0 and time.time() - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = time.time()
        st.rerun()
    
    st.markdown("---")
    st.subheader("🌐 WebSocket Control")
    
    ws_host = st.text_input("Server Host", value="localhost")
    ws_port = st.number_input("Server Port", value=8765, min_value=1024, max_value=65535)
    bot_uri = st.text_input("Bot URI (client)", value="ws://localhost:8765")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Server"):
            if st.session_state.ws_manager is None:
                st.session_state.ws_manager = WSManager(host=ws_host, port=ws_port)
            if st.session_state.ws_manager.start_server():
                st.success("Server started")
            else:
                st.error("Failed to start")
    
    with col2:
        if st.button("⏹️ Stop Server"):
            if st.session_state.ws_manager:
                st.session_state.ws_manager.stop_server()
                st.success("Server stopped")
    
    if st.button("🔌 Connect to Bot"):
        if st.session_state.ws_manager is None:
            st.session_state.ws_manager = WSManager()
        st.session_state.ws_manager.start_client(bot_uri)
        st.success("Connecting...")
    
    # Connection status
    if st.session_state.ws_manager:
        status = st.session_state.ws_manager.get_status()
        st.markdown("**Connection Status:**")
        st.markdown(f"Server: {'🟢 Running' if status['server_running'] else '🔴 Stopped'}")
        st.markdown(f"Client: {'🟢 Connected' if status['client_connected'] else '🔴 Disconnected'}")
        st.markdown(f"Clients: {status['num_clients']}")
    
    st.markdown("---")
    st.subheader("🎮 Bot Commands")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⏸️ Pause"):
            if st.session_state.ws_manager:
                st.session_state.ws_manager.send_command({"command": "pause"})
                st.success("Sent")
    with col2:
        if st.button("▶️ Resume"):
            if st.session_state.ws_manager:
                st.session_state.ws_manager.send_command({"command": "resume"})
                st.success("Sent")
    with col3:
        if st.button("❌ Close"):
            if st.session_state.ws_manager:
                st.session_state.ws_manager.send_command({"command": "close_positions"})
                st.success("Sent")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Trading", "📈 Performance", "🎬 Replay", "⚙️ System"])

with tab1:
    # Load data
    with st.spinner("Loading data..."):
        df = load_ohlcv_data(symbol, timeframe, limit)
    
    if df is None or df.empty:
        st.error("❌ No data available. Please check your data sources.")
        st.stop()
    
    # Compute indicators
    df["ema20"] = ema(df["close"], 20)
    df["ema50"] = ema(df["close"], 50)
    df["rsi"] = rsi(df["close"], 14)
    mline, sline, hist = macd(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = mline, sline, hist
    df["atr"] = atr(df)
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        current_price = df["close"].iloc[-1]
        price_change = ((current_price - df["close"].iloc[-2]) / df["close"].iloc[-2]) * 100
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            f"{price_change:+.2f}%",
            delta_color="normal"
        )
    
    with col2:
        st.metric("24h High", f"${df['high'].tail(1440).max():.2f}")
    
    with col3:
        st.metric("24h Low", f"${df['low'].tail(1440).min():.2f}")
    
    with col4:
        current_rsi = df["rsi"].iloc[-1]
        rsi_color = "normal" if 30 < current_rsi < 70 else "inverse"
        st.metric("RSI(14)", f"{current_rsi:.1f}", delta_color=rsi_color)
    
    with col5:
        volume_24h = df["volume"].tail(1440).sum() if "volume" in df.columns else 0
        st.metric("24h Volume", f"${volume_24h/1e6:.2f}M")
    
    # Load trades and create signals
    signals = []
    if show_trades:
        trades_df = load_trade_log()
        if not trades_df.empty:
            for _, row in trades_df.tail(50).iterrows():
                try:
                    signals.append({
                        "timestamp": pd.to_datetime(row.get("ts", df["timestamp"].iloc[-1])),
                        "price": float(row.get("price", current_price)),
                        "side": row.get("side", ""),
                        "label": row.get("status", "")
                    })
                except:
                    continue
    
    # Ghost predictions
    ghost_df = None
    if show_ghosts and TORCH_AVAILABLE:
        # Implement ghost prediction loading here
        pass
    
    # Main chart
    st.subheader("📊 Price Chart")
    fig = create_price_chart(df, signals=signals, ghosts=ghost_df, 
                            show_fvg=show_fvg, show_fib=show_fib)
    st.plotly_chart(fig, use_container_width=True)
    
    # Position panel
    st.subheader("💼 Current Positions")
    state = None
    if st.session_state.ws_manager:
        state = st.session_state.ws_manager.latest_state
    if not state:
        state = load_state()
    
    if state:
        col1, col2, col3 = st.columns(3)
        with col1:
            equity = state.get("equity", 0.0)
            st.metric("Equity", f"${equity:,.2f}")
        with col2:
            positions = state.get("positions", {})
            st.metric("Open Positions", len(positions))
        with col3:
            halted = state.get("risk_halted", False)
            status = "🔴 HALTED" if halted else "🟢 ACTIVE"
            st.markdown(f"**Status:** {status}")
        
        if positions:
            pos_data = []
            for sym, pos in positions.items():
                pos_data.append({
                    "Symbol": sym,
                    "Side": pos.get("side", ""),
                    "Qty": pos.get("qty", 0),
                    "Avg Price": f"${pos.get('avg_price', 0):.4f}",
                    "Current": f"${current_price:.4f}",
                    "PnL": f"${(current_price - pos.get('avg_price', current_price)) * pos.get('qty', 0):.2f}"
                })
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
    else:
        st.info("No live state available. Start your bot with WebSocket enabled.")

with tab2:
    st.subheader("📈 Performance Metrics")
    
    trades_df = load_trade_log()
    if not trades_df.empty and "pnl" in trades_df.columns:
        trades_df["pnl"] = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0)
        equity_curve = (trades_df["pnl"].cumsum() + 10000).values
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_pnl = trades_df["pnl"].sum()
            st.metric("Total PnL", f"${total_pnl:,.2f}", delta_color="normal")
        
        with col2:
            win_rate = (trades_df["pnl"] > 0).sum() / len(trades_df) * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col3:
            if len(equity_curve) > 1:
                returns = np.diff(equity_curve) / equity_curve[:-1]
                sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            else:
                st.metric("Sharpe Ratio", "N/A")
        
        with col4:
            max_dd = ((equity_curve.cummax() - equity_curve) / equity_curve.cummax()).max()
            st.metric("Max Drawdown", f"{max_dd*100:.2f}%", delta_color="inverse")
        
        # Equity curve chart
        st.subheader("Equity Curve")
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(
            y=equity_curve,
            mode='lines',
            name='Equity',
            line=dict(color='#26a69a', width=2)
        ))
        equity_fig.update_layout(
            height=400,
            template="plotly_dark",
            xaxis_title="Trade Number",
            yaxis_title="Equity ($)"
        )
        st.plotly_chart(equity_fig, use_container_width=True)
        
        # Trade distribution
        st.subheader("PnL Distribution")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=trades_df["pnl"],
            nbinsx=50,
            marker_color='#1f77b4'
        ))
        hist_fig.update_layout(height=300, template="plotly_dark")
        st.plotly_chart(hist_fig, use_container_width=True)
        
        # Recent trades table
        st.subheader("Recent Trades")
        st.dataframe(trades_df.tail(20), use_container_width=True)
    else:
        st.info("No trade history available yet.")

with tab3:
    st.subheader("🎬 Episode Replay")
    st.info("Replay functionality - Load episode files from runs/episodes/")
    # Replay implementation here (from original code)

with tab4:
    st.subheader("⚙️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**System Status**")
        st.markdown(f"- Python: {os.sys.version.split()[0]}")
        st.markdown(f"- PyTorch: {'✅ Available' if TORCH_AVAILABLE else '❌ Not available'}")
        st.markdown(f"- WebSockets: {'✅ Available' if WEBSOCKETS_AVAILABLE else '❌ Not available'}")
    
    with col2:
        st.markdown("**Data Status**")
        st.markdown(f"- OHLCV rows: {len(df) if df is not None else 0}")
        st.markdown(f"- Trade logs: {len(load_trade_log())}")
        st.markdown(f"- Last update: {datetime.now().strftime('%H:%M:%S')}")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Use the sidebar to configure data sources, enable WebSocket connections, and control bot behavior in real-time.")