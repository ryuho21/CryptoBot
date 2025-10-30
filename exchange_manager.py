# exchange_manager.py - PRODUCTION-READY EXCHANGE MANAGER
"""
Complete exchange manager with:
- Order execution
- Position tracking
- Risk management
- Rate limiting
- Timeout handling
- Position reconciliation
"""

import ccxt
import time
import numpy as np
from collections import deque
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Production-ready risk management with:
    - Drawdown monitoring
    - Position limits
    - Trade throttling
    - Circuit breaker
    """

    def __init__(self, config: Any):
        self.config = config
        self.net_worth = config.initial_balance
        self.max_net_worth = config.initial_balance
        self.max_drawdown = 0.0
        self.circuit_breaker_tripped = False

        # Trade throttling
        self.trade_history = deque(maxlen=config.max_trades_per_minute)
        self.last_trade_time = 0.0

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0

    def update_metrics(self, new_net_worth: float):
        """Update risk metrics with new net worth."""
        self.net_worth = new_net_worth
        self.max_net_worth = max(self.max_net_worth, new_net_worth)

        # Calculate drawdown
        current_dd = (self.max_net_worth - new_net_worth) / (self.max_net_worth + 1e-8)
        self.max_drawdown = max(self.max_drawdown, current_dd)

        # Circuit breaker
        if current_dd >= self.config.max_drawdown_pct and not self.circuit_breaker_tripped:
            self.circuit_breaker_tripped = True
            logger.critical(f"🚨 CIRCUIT BREAKER TRIPPED: Drawdown {current_dd:.2%}")

    def calculate_position_size_usd(self, market_price: float, current_position_value: float,
                                    kelly_fraction: Optional[float] = None) -> float:
        """
        Calculate position size using Kelly Criterion.

        Args:
            market_price: current market price
            current_position_value: value of current position
            kelly_fraction: fraction of Kelly to use

        Returns:
            position_size_usd: USD value of position to take
        """
        if kelly_fraction is None:
            kelly_fraction = self.config.kelly_criterion_fraction

        # Simple Kelly approximation (can be improved with win rate estimation)
        win_rate = 0.55 if self.total_trades == 0 else self.winning_trades / self.total_trades
        avg_gain = 0.015
        avg_loss = 0.010

        p = max(0.5, min(0.7, win_rate))  # Clamp to reasonable range
        q = 1.0 - p

        if avg_loss <= 0:
            kelly_f = 0.0
        else:
            kelly_f = p - (q / (avg_gain / avg_loss))

        if kelly_f <= 0:
            return 0.0

        # Apply fractional Kelly
        size_fraction = kelly_f * kelly_fraction
        size_fraction = min(size_fraction, self.config.max_position_size)

        # Available capital (leave 5% buffer)
        available_capital = self.net_worth * 0.95
        size_usd = available_capital * size_fraction

        # Minimum trade size
        if size_usd < 100.0:
            return 0.0

        # Max position with leverage
        max_position_usd = self.net_worth * self.config.max_position_leverage
        size_usd = min(size_usd, max_position_usd)

        return max(0.0, size_usd)

    def check_trade_throttling(self) -> Tuple[bool, str]:
        """Check if trade is allowed based on throttling rules."""
        now = time.time()

        # Time-based throttle
        if now - self.last_trade_time < self.config.trade_throttle_seconds:
            return False, f"Throttled: Wait {self.config.trade_throttle_seconds}s between trades"

        # Rate limit (trades per minute)
        one_minute_ago = now - 60
        recent_trades = [t for t in self.trade_history if t > one_minute_ago]

        if len(recent_trades) >= self.config.max_trades_per_minute:
            return False, f"Rate limit: {len(recent_trades)}/{self.config.max_trades_per_minute} trades/min"

        return True, ""

    def log_trade(self, pnl: Optional[float] = None):
        """Log trade execution."""
        self.last_trade_time = time.time()
        self.trade_history.append(self.last_trade_time)
        self.total_trades += 1

        if pnl is not None:
            self.total_pnl += pnl
            if pnl > 0:
                self.winning_trades += 1

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker (use with caution)."""
        self.circuit_breaker_tripped = False
        logger.warning("⚠️ Circuit breaker manually reset")

    def get_statistics(self) -> Dict:
        """Return risk statistics."""
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0

        return {
            'net_worth': self.net_worth,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'circuit_breaker': self.circuit_breaker_tripped
        }


class ExchangeManager:
    """
    Production-ready exchange manager with:
    - Real order execution
    - Position reconciliation
    - Timeout handling
    - Paper trading mode
    """

    def __init__(self, risk_manager: RiskManager, config: Any):
        self.risk_manager = risk_manager
        self.config = config
        self.current_position = 0.0  # Contracts (base units)
        self.position_entry_price = 0.0
        self.max_retries = config.max_retries

        # Initialize exchange
        self.exchange = None
        self._init_exchange()

        # Position reconciliation
        self.last_reconciliation = time.time()
        self.reconciliation_interval = 300  # 5 minutes

    def _init_exchange(self):
        """Initialize CCXT exchange."""
        try:
            exchange_config = {
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'password': self.config.api_passphrase,
                'enableRateLimit': True,
                'timeout': self.config.order_timeout_seconds * 1000,
                'options': {'defaultType': 'swap'}
            }

            # instantiate exchange object
            self.exchange = ccxt.okx(exchange_config)

            if self.config.use_testnet:
                try:
                    # not all CCXT builds support set_sandbox_mode; guard it
                    if hasattr(self.exchange, "set_sandbox_mode"):
                        self.exchange.set_sandbox_mode(True)
                    logger.info("✅ OKX Sandbox Mode Enabled")
                except Exception:
                    logger.warning("⚠️ Sandbox mode not available on this CCXT build")

            if self.config.paper_trade:
                logger.info("📄 Paper trading mode enabled")
            else:
                # Attempt to load real markets
                try:
                    self.exchange.load_markets()
                    logger.info("✅ Real markets loaded")
                except Exception as e:
                    logger.error(f"Failed to load markets: {e}")
                    # If loading fails, keep exchange object but warn
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")
            # Leave self.exchange = None for pure paper/offline operation
            self.exchange = None

    def _load_markets(self):
        """Load market information into exchange.markets (mock if needed)."""
        # If exchange object exists, try to load markets
        if self.exchange is not None:
            try:
                self.exchange.load_markets()
                logger.info("✅ Real markets loaded")
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to load real markets: {e}")

        # Create mock markets structure for paper trading or if exchange is None
        mock_market = {
            'symbol': self.config.symbol,
            'id': self.config.symbol.replace('/', '-'),
            'base': 'BTC',
            'quote': 'USDT',
            'type': 'swap',
            'contractSize': 0.01,
            'precision': {'amount': 6, 'price': 2},
            'limits': {
                'amount': {'min': 0.0001, 'max': 1e6},
                'price': {'min': 0.01, 'max': 1e9},
                'cost': {'min': 1}
            }
        }

        # Ensure exchange has a markets dict to read from
        if self.exchange is None:
            class _MockExchange:
                pass
            self.exchange = _MockExchange()

        # Attach mock markets
        try:
            setattr(self.exchange, "markets", {self.config.symbol: mock_market})
            logger.info("✅ Mock markets loaded into exchange")
        except Exception as e:
            logger.warning(f"⚠️ Could not attach mock markets: {e}")

    def reconcile_position(self):
        """Reconcile local position with exchange (for live trading)."""
        if self.config.paper_trade or self.exchange is None:
            return

        now = time.time()
        if now - self.last_reconciliation < self.reconciliation_interval:
            return

        try:
            positions = self.exchange.fetch_positions([self.config.symbol])

            if positions:
                exchange_position = float(positions[0].get('contracts', 0))

                if not np.isclose(self.current_position, exchange_position, atol=0.1):
                    logger.warning(
                        f"⚠️ Position drift detected: "
                        f"Local={self.current_position}, Exchange={exchange_position}"
                    )
                    self.current_position = exchange_position

            self.last_reconciliation = now

        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")

    def execute_order(self, action_type: str, price: float, size_usd: float) -> Tuple[bool, Any]:
        """
        Execute order with proper error handling.

        Args:
            action_type: 'BUY', 'SELL', or 'HOLD'
            price: current market price
            size_usd: USD value to trade

        Returns:
            Tuple of (success: bool, message/order dict)
        """
        # Check circuit breaker
        if self.risk_manager.circuit_breaker_tripped:
            return False, "Circuit breaker tripped"

        # Check throttling
        can_trade, reason = self.risk_manager.check_trade_throttling()
        if not can_trade:
            return False, reason

        # Handle HOLD
        if action_type == 'HOLD':
            return True, "No action"

        # Check minimum size
        if np.isclose(size_usd, 0.0) or size_usd < 10.0:
            return True, "Order too small, skipped"

        # Paper trading mode
        if self.config.paper_trade:
            return self._execute_paper_order(action_type, price, size_usd)

        # Real order execution
        return self._execute_real_order(action_type, price, size_usd)

    def _execute_paper_order(self, action_type: str, price: float, size_usd: float) -> Tuple[bool, Dict]:
        """
        Simulate order execution in paper trading mode.
        Provides fallback even when no CCXT exchange instance is loaded.
        Returns (success: bool, result: dict)
        """
        try:
            # Ensure markets metadata available (fallback to mock)
            market = None
            if self.exchange is not None and hasattr(self.exchange, "markets"):
                try:
                    market = getattr(self.exchange, "markets", {}).get(self.config.symbol)
                except Exception:
                    market = None

            # Defaults if metadata missing
            contract_size = 1.0
            price_precision = 2
            amount_precision = 6

            if market:
                contract_size = market.get('contractSize', contract_size)
                precision = market.get('precision', {})
                price_precision = precision.get('price', price_precision)
                amount_precision = precision.get('amount', amount_precision)

            # Calculate number of contracts (base units)
            # size_usd / (price * contract_size) gives contracts count
            num_contracts = size_usd / (price * contract_size)
            if num_contracts <= 0:
                return True, {"status": "skipped", "reason": "zero_size"}

            # Keep as float (allow fractional contracts if meaningful), round to amount_precision
            factor = 10 ** amount_precision
            num_contracts = float(int(num_contracts * factor)) / factor
            if num_contracts == 0:
                return True, {"status": "skipped", "reason": "rounded_to_zero"}

            # Simulate slippage and spread
            slippage = price * self.config.slippage_pct
            spread = price * self.config.spread_pct

            if action_type.upper() == 'BUY':
                exec_price = price + slippage + spread
                # If closing short
                if self.current_position < 0:
                    close_amount = min(num_contracts, abs(self.current_position))
                    # PnL: (entry - exec_price) * close_amount * contract_size for short close
                    pnl = (self.position_entry_price - exec_price) * close_amount * contract_size
                    self.current_position += close_amount
                    self.risk_manager.log_trade(pnl)
                    result = {"status": "filled", "side": "buy", "amount": close_amount, "price": exec_price, "pnl": pnl}
                    logger.info(f"📄 Paper CLOSE SHORT: {close_amount} @ {exec_price:.2f} | PnL: ${pnl:.2f}")
                else:
                    # Opening/increasing long
                    self.current_position += num_contracts
                    self.position_entry_price = exec_price
                    self.risk_manager.log_trade(None)
                    result = {"status": "filled", "side": "buy", "amount": num_contracts, "price": exec_price}
                    logger.info(f"📄 Paper BUY: {num_contracts} @ {exec_price:.2f}")

            elif action_type.upper() == 'SELL':
                exec_price = price - slippage - spread
                # If closing long
                if self.current_position > 0:
                    close_amount = min(num_contracts, abs(self.current_position))
                    pnl = (exec_price - self.position_entry_price) * close_amount * contract_size
                    self.current_position -= close_amount
                    self.risk_manager.log_trade(pnl)
                    result = {"status": "filled", "side": "sell", "amount": close_amount, "price": exec_price, "pnl": pnl}
                    logger.info(f"📄 Paper CLOSE LONG: {close_amount} @ {exec_price:.2f} | PnL: ${pnl:.2f}")
                else:
                    # Opening/increasing short
                    self.current_position -= num_contracts
                    self.position_entry_price = exec_price
                    self.risk_manager.log_trade(None)
                    result = {"status": "filled", "side": "sell", "amount": num_contracts, "price": exec_price}
                    logger.info(f"📄 Paper SELL: {num_contracts} @ {exec_price:.2f}")
            else:
                return True, {"status": "no_action"}

            # Normalize tiny floating noise
            self.current_position = float(round(self.current_position, 8))

            return True, result

        except Exception as e:
            logger.warning(f"⚠️ Paper order execution failed: {e}")
            return False, {"status": "failed", "error": str(e)}

    def _execute_real_order(self, action_type: str, price: float, size_usd: float) -> Tuple[bool, Any]:
        """Execute real exchange order."""
        # Ensure markets loaded (attempt if not)
        if not hasattr(self.exchange, "markets") or self.exchange.markets is None:
            try:
                self._load_markets()
            except Exception:
                logger.warning("⚠️ Markets missing; proceeding with best-effort execution")

        # obtain market metadata with safe fallbacks
        market = None
        try:
            market = getattr(self.exchange, "markets", {}).get(self.config.symbol)
        except Exception:
            market = None

        contract_size = 1.0
        min_amount = 0.0
        precision_amount = 6

        if market:
            contract_size = market.get('contractSize', contract_size)
            precision_amount = market.get('precision', {}).get('amount', precision_amount)
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0.0)

        num_contracts = size_usd / (price * contract_size)
        # round to precision
        factor = 10 ** precision_amount
        num_contracts = float(int(num_contracts * factor)) / factor

        if num_contracts < min_amount:
            return True, "Below minimum size"

        try:
            amount = float(self.exchange.amount_to_precision(self.config.symbol, num_contracts))
        except Exception:
            amount = float(num_contracts)

        # Determine side and parameters
        side = None
        params = {}

        if action_type.upper() == 'BUY':
            side = 'buy'
            if self.current_position < 0:
                params = {'reduceOnly': True}
                amount = min(amount, abs(self.current_position))
        elif action_type.upper() == 'SELL':
            side = 'sell'
            if self.current_position > 0:
                params = {'reduceOnly': True}
                amount = min(amount, abs(self.current_position))
        else:
            return True, "No action"

        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                # Many CCXT exchanges expect (symbol, type, side, amount, price, params) for certain methods,
                # but create_market_order often takes (symbol, side, amount, params)
                order = self.exchange.create_market_order(self.config.symbol, side, amount, params)

                # Update position
                self._update_local_position(order)
                self.risk_manager.log_trade(None)

                logger.info(f"✅ Order executed: {side.upper()} {amount} @ market")
                return True, order

            except ccxt.DDoSProtection as e:
                logger.warning(f"Rate limit hit (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)

            except ccxt.InvalidOrder as e:
                logger.error(f"Invalid order: {e}")
                return False, str(e)

            except Exception as e:
                logger.error(f"Order failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return False, str(e)
                time.sleep(1)

        return False, "Execution failed after retries"

    def _update_local_position(self, order: Dict):
        """Update local position tracking from order."""
        try:
            filled_amount = float(order.get('filled', 0))
        except Exception:
            # Some exchanges return different fields; attempt to extract
            try:
                filled_amount = float(order.get('amount', 0))
            except Exception:
                filled_amount = 0.0

        side = order.get('side', '').lower()

        if side == 'buy':
            self.current_position += filled_amount
        elif side == 'sell':
            self.current_position -= filled_amount

        # Round to avoid float precision issues
        self.current_position = round(self.current_position, 8)
