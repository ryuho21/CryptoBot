# notification_service.py - PRODUCTION-READY NOTIFICATIONS
"""
Complete notification service with:
- Telegram integration
- Discord webhooks
- Email alerts (optional)
- Rate limiting
- Priority levels
"""

import requests
import time
from typing import Optional, Dict
from collections import deque
import logging

logger = logging.getLogger(__name__)


class NotificationService:
    """
    Multi-channel notification service with rate limiting.
    
    Supports:
    - Telegram
    - Discord
    - Console logging
    """
    
    def __init__(self, config):
        self.config = config
        
        # Telegram
        self.telegram_token = config.telegram_token
        self.telegram_chat_id = config.telegram_chat_id
        self.telegram_enabled = bool(self.telegram_token and self.telegram_chat_id)
        
        # Discord
        self.discord_webhook = config.discord_webhook
        self.discord_enabled = bool(self.discord_webhook)
        
        # Rate limiting
        self.message_history = deque(maxlen=100)
        self.rate_limit_window = 60  # seconds
        self.max_messages_per_window = 10
        
        logger.info(f"📢 Notifications: Telegram={'✅' if self.telegram_enabled else '❌'}, "
                   f"Discord={'✅' if self.discord_enabled else '❌'}")
    
    def send(self, message: str, priority: str = 'INFO', silent: bool = False):
        """
        Send notification to all enabled channels.
        
        Args:
            message: notification text
            priority: 'INFO', 'WARNING', 'CRITICAL'
            silent: suppress notification sound
        """
        # Rate limit check
        if not self._check_rate_limit():
            logger.warning(f"⚠️ Notification rate limit exceeded, skipping: {message[:50]}...")
            return
        
        # Format message with priority
        formatted_msg = f"[{priority}] {message}"
        
        # Log to console
        if priority == 'CRITICAL':
            logger.critical(formatted_msg)
        elif priority == 'WARNING':
            logger.warning(formatted_msg)
        else:
            logger.info(formatted_msg)
        
        # Send to channels
        if self.telegram_enabled:
            self._send_telegram(formatted_msg, silent)
        
        if self.discord_enabled:
            self._send_discord(formatted_msg, priority)
        
        # Record send time
        self.message_history.append(time.time())
    
    def send_chart(self, image_path: str, caption: str = ""):
        """Send chart image (Telegram only)."""
        if not self.telegram_enabled:
            return
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
            
            with open(image_path, 'rb') as img:
                files = {'photo': img}
                data = {
                    'chat_id': self.telegram_chat_id,
                    'caption': caption
                }
                
                response = requests.post(url, files=files, data=data, timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"✅ Chart sent: {image_path}")
                else:
                    logger.warning(f"⚠️ Chart send failed: {response.text}")
        
        except Exception as e:
            logger.error(f"❌ Chart send error: {e}")
    
    def _send_telegram(self, message: str, silent: bool = False):
        """Send message via Telegram."""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_notification': silent
            }
            
            response = requests.post(url, json=data, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Telegram send failed: {response.text}")
        
        except Exception as e:
            logger.error(f"Telegram error: {e}")
    
    def _send_discord(self, message: str, priority: str):
        """Send message via Discord webhook."""
        try:
            # Color based on priority
            color_map = {
                'INFO': 3447003,      # Blue
                'WARNING': 16776960,  # Yellow
                'CRITICAL': 15158332  # Red
            }
            
            data = {
                'embeds': [{
                    'title': f'Trading Bot - {priority}',
                    'description': message,
                    'color': color_map.get(priority, 3447003)
                }]
            }
            
            response = requests.post(self.discord_webhook, json=data, timeout=10)
            
            if response.status_code not in [200, 204]:
                logger.warning(f"Discord send failed: {response.text}")
        
        except Exception as e:
            logger.error(f"Discord error: {e}")
    
    def _check_rate_limit(self) -> bool:
        """Check if sending is allowed based on rate limit."""
        now = time.time()
        window_start = now - self.rate_limit_window
        
        # Count recent messages
        recent_messages = sum(1 for t in self.message_history if t > window_start)
        
        return recent_messages < self.max_messages_per_window
    
    def send_update(self, message: str):
        """Alias for send() with INFO priority."""
        self.send(message, priority='INFO')
    
    def send_warning(self, message: str):
        """Send warning notification."""
        self.send(message, priority='WARNING')
    
    def send_critical(self, message: str):
        """Send critical notification (no rate limit)."""
        formatted_msg = f"[CRITICAL] {message}"
        logger.critical(formatted_msg)
        
        if self.telegram_enabled:
            self._send_telegram(formatted_msg, silent=False)
        
        if self.discord_enabled:
            self._send_discord(formatted_msg, 'CRITICAL')