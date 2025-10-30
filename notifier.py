"""
notifier.py - IMPROVED VERSION

Improvements:
1. Added retry logic for failed requests
2. Rate limiting to prevent API spam
3. Message queue for high-frequency events
4. Error rate tracking and circuit breaker
5. Better error handling and logging
6. HTML formatting support for Telegram
7. Message batching for efficiency
"""

import os
import json
import logging
import requests
import threading
import time
from datetime import datetime
from collections import deque
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


class RateLimiter:
    """Simple token bucket rate limiter."""
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = threading.Lock()
    
    def can_send(self) -> bool:
        with self.lock:
            now = time.time()
            # Remove old calls outside the window
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            return False
    
    def wait_if_needed(self):
        """Block until rate limit allows sending."""
        while not self.can_send():
            time.sleep(0.1)


class MessageQueue:
    """Queue for batching and throttling messages."""
    def __init__(self, flush_interval: float = 5.0, max_batch_size: int = 10):
        self.queue = deque()
        self.flush_interval = flush_interval
        self.max_batch_size = max_batch_size
        self.lock = threading.Lock()
        self.last_flush = time.time()
    
    def add(self, message: str):
        with self.lock:
            self.queue.append(message)
    
    def should_flush(self) -> bool:
        with self.lock:
            return (len(self.queue) >= self.max_batch_size or 
                    time.time() - self.last_flush >= self.flush_interval)
    
    def get_batch(self) -> List[str]:
        with self.lock:
            batch = list(self.queue)
            self.queue.clear()
            self.last_flush = time.time()
            return batch


class Notifier:
    """
    Unified Notification Manager with improved reliability and features.
    """

    def __init__(self, config=None):
        # Environment variables
        self.telegram_token = os.environ.get("TELEGRAM_TOKEN", "")
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        self.discord_webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")

        self.enable_telegram = bool(self.telegram_token and self.telegram_chat_id)
        self.enable_discord = bool(self.discord_webhook)

        if not self.enable_telegram and not self.enable_discord:
            logger.warning("Notifier: No active channels (Telegram or Discord). Alerts disabled.")
        else:
            logger.info("Notifier initialized: Telegram=%s, Discord=%s", 
                       self.enable_telegram, self.enable_discord)

        # Rate limiting (30 messages per minute for Telegram API)
        self.telegram_limiter = RateLimiter(max_calls=30, period=60.0)
        self.discord_limiter = RateLimiter(max_calls=50, period=60.0)
        
        # Message queue for batching
        self.message_queue = MessageQueue(flush_interval=5.0, max_batch_size=5)
        
        # Error tracking for circuit breaker
        self.error_count = {'telegram': 0, 'discord': 0}
        self.error_threshold = 5
        self.error_reset_time = {'telegram': time.time(), 'discord': time.time()}
        self.circuit_open = {'telegram': False, 'discord': False}
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 2.0  # seconds
        
        self._lock = threading.Lock()
        
        # Start background flush thread
        self._flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self._flush_thread.start()

    # --------------------------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------------------------
    def send(self, text: str, parse_mode: str = "Markdown", silent: bool = False, 
             priority: str = "normal"):
        """
        Send a message to all enabled channels.
        
        Args:
            text: Message text
            parse_mode: "Markdown" or "HTML"
            silent: Disable notification sound
            priority: "high" for immediate send, "normal" for queuing
        """
        text = text.strip()
        if not text:
            return

        # Format message
        msg = self._format_message(text)
        
        if priority == "high":
            # Send immediately
            self._send_immediate(msg, parse_mode, silent)
        else:
            # Add to queue for batching
            self.message_queue.add(msg)

    def trade_alert(self, symbol: str, side: str, amount: float, price: float, 
                   tp: Optional[float] = None, sl: Optional[float] = None, 
                   pnl: Optional[float] = None):
        """Send formatted trade entry/exit alert."""
        side_lower = side.lower()
        
        if pnl is not None:
            # Exit trade
            emoji = "✅" if pnl > 0 else "❌"
            msg = (
                f"{emoji} *TRADE CLOSED*\n"
                f"Symbol: `{symbol}`\n"
                f"Side: `{side.upper()}`\n"
                f"Exit Price: `{price:.4f}`\n"
                f"PnL: `{pnl:+.6f}` ({pnl/price*100:+.2f}%)"
            )
        else:
            # Entry trade
            side_emoji = "🟩 LONG" if side_lower in ("buy", "long") else "🟥 SHORT"
            msg = (
                f"⚡️ *TRADE EXECUTED*\n"
                f"{side_emoji}\n"
                f"Symbol: `{symbol}`\n"
                f"Amount: `{amount:.6f}`\n"
                f"Entry: `{price:.4f}`"
            )
            if tp or sl:
                msg += "\n"
            if tp:
                msg += f"🎯 TP: `{tp:.4f}`\n"
            if sl:
                msg += f"🛑 SL: `{sl:.4f}`"

        self.send(msg, priority="high")

    def risk_alert(self, alert_type: str, detail: str = "", metrics: Dict = None):
        """Send risk management alerts (drawdown, circuit breaker, etc.)."""
        emoji_map = {
            'drawdown': '📉',
            'circuit_breaker': '🚨',
            'position_limit': '⚠️',
            'margin_call': '🔴',
        }
        emoji = emoji_map.get(alert_type, '⚠️')
        
        msg = f"{emoji} *RISK ALERT: {alert_type.upper()}*\n{detail}"
        
        if metrics:
            msg += "\n\n*Metrics:*"
            for k, v in metrics.items():
                msg += f"\n- {k}: `{v}`"
        
        self.send(msg, priority="high")

    def system_alert(self, title: str, detail: str = "", level: str = "warning"):
        """Send system-wide error or warning."""
        emoji_map = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }
        emoji = emoji_map.get(level, '⚠️')
        msg = f"{emoji} *{title}*\n{detail}"
        priority = "high" if level in ("error", "critical") else "normal"
        self.send(msg, priority=priority)

    def performance_report(self, metrics: Dict, period: str = "Daily"):
        """Send performance summary report."""
        msg = f"📊 *{period.upper()} SUMMARY*\n\n"
        
        # Format metrics nicely
        for k, v in metrics.items():
            if isinstance(v, float):
                if 'pct' in k.lower() or 'rate' in k.lower():
                    msg += f"• {k}: `{v:.2f}%`\n"
                elif 'sharpe' in k.lower() or 'ratio' in k.lower():
                    msg += f"• {k}: `{v:.3f}`\n"
                else:
                    msg += f"• {k}: `{v:.4f}`\n"
            else:
                msg += f"• {k}: `{v}`\n"
        
        self.send(msg)

    def shutdown(self):
        """Flush remaining messages and clean shutdown."""
        logger.info("Notifier shutting down, flushing queue...")
        batch = self.message_queue.get_batch()
        if batch:
            self._send_batch(batch)
        logger.info("Notifier shutdown complete")

    # --------------------------------------------------------------------------
    # PRIVATE METHODS
    # --------------------------------------------------------------------------
    def _format_message(self, text: str) -> str:
        """Add timestamp and formatting to message."""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"{text}\n\n⏱ *{now}*"

    def _send_immediate(self, message: str, parse_mode: str, silent: bool):
        """Send message immediately without queuing."""
        threads = []
        if self.enable_telegram and not self.circuit_open['telegram']:
            t = threading.Thread(
                target=self._send_telegram_with_retry,
                args=(message, parse_mode, silent),
                daemon=True
            )
            threads.append(t)
            t.start()
        
        if self.enable_discord and not self.circuit_open['discord']:
            t = threading.Thread(
                target=self._send_discord_with_retry,
                args=(message,),
                daemon=True
            )
            threads.append(t)
            t.start()
        
        # Optionally wait for threads
        for t in threads:
            t.join(timeout=5.0)

    def _background_flush(self):
        """Background thread to flush message queue periodically."""
        while True:
            try:
                time.sleep(1.0)
                if self.message_queue.should_flush():
                    batch = self.message_queue.get_batch()
                    if batch:
                        self._send_batch(batch)
            except Exception as e:
                logger.exception("Error in background flush: %s", e)

    def _send_batch(self, messages: List[str]):
        """Send a batch of messages as a single combined message."""
        if not messages:
            return
        
        combined = "\n\n---\n\n".join(messages)
        # Telegram has 4096 char limit
        if len(combined) > 4000:
            # Split into chunks
            for msg in messages:
                self._send_immediate(msg, "Markdown", False)
        else:
            self._send_immediate(combined, "Markdown", False)

    def _send_telegram_with_retry(self, message: str, parse_mode: str = "Markdown", 
                                  silent: bool = False):
        """Send to Telegram with retry logic and circuit breaker."""
        if self.circuit_open['telegram']:
            # Check if circuit should reset (5 minute timeout)
            if time.time() - self.error_reset_time['telegram'] > 300:
                self.circuit_open['telegram'] = False
                self.error_count['telegram'] = 0
                logger.info("Telegram circuit breaker reset")
            else:
                logger.warning("Telegram circuit breaker is open, skipping send")
                return
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                self.telegram_limiter.wait_if_needed()
                
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                payload = {
                    "chat_id": self.telegram_chat_id,
                    "text": message,
                    "parse_mode": parse_mode,
                    "disable_notification": silent,
                }
                
                resp = requests.post(url, data=payload, timeout=10)
                
                if resp.status_code == 200:
                    # Success - reset error count
                    self.error_count['telegram'] = 0
                    return
                elif resp.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(resp.headers.get('Retry-After', 5))
                    logger.warning(f"Telegram rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                else:
                    logger.error(f"Telegram send failed: {resp.status_code} - {resp.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Telegram timeout (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.exception(f"Telegram error (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        # All retries failed
        self.error_count['telegram'] += 1
        if self.error_count['telegram'] >= self.error_threshold:
            self.circuit_open['telegram'] = True
            self.error_reset_time['telegram'] = time.time()
            logger.error("Telegram circuit breaker opened after repeated failures")

    def _send_discord_with_retry(self, message: str):
        """Send to Discord with retry logic and circuit breaker."""
        if self.circuit_open['discord']:
            if time.time() - self.error_reset_time['discord'] > 300:
                self.circuit_open['discord'] = False
                self.error_count['discord'] = 0
                logger.info("Discord circuit breaker reset")
            else:
                return
        
        for attempt in range(self.max_retries):
            try:
                self.discord_limiter.wait_if_needed()
                
                data = {"content": message[:2000]}  # Discord limit is 2000 chars
                headers = {"Content-Type": "application/json"}
                
                resp = requests.post(
                    self.discord_webhook, 
                    data=json.dumps(data), 
                    headers=headers, 
                    timeout=10
                )
                
                if resp.status_code in (200, 204):
                    self.error_count['discord'] = 0
                    return
                else:
                    logger.error(f"Discord send failed: {resp.status_code} - {resp.text}")
                    
            except Exception as e:
                logger.exception(f"Discord error (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        # All retries failed
        self.error_count['discord'] += 1
        if self.error_count['discord'] >= self.error_threshold:
            self.circuit_open['discord'] = True
            self.error_reset_time['discord'] = time.time()
            logger.error("Discord circuit breaker opened after repeated failures")