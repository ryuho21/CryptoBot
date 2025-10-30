# logger.py - PRODUCTION-READY LOGGING
"""
Global structured logger using Loguru.
Provides async-safe, colorized logging across all modules.
"""

from loguru import logger
import sys
import os
from datetime import datetime

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Generate log filename with timestamp
log_file = f"logs/trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logger
logger.remove()  # Remove default handler

# Console handler (colorized)
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# File handler (detailed)
logger.add(
    log_file,
    rotation="50 MB",
    retention="30 days",
    level="DEBUG",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# Error file (separate)
logger.add(
    "logs/errors.log",
    rotation="10 MB",
    retention="90 days",
    level="ERROR",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}"
)

# Export logger
LOGGER = logger

# Log startup
LOGGER.info("="*70)
LOGGER.info("🤖 Trading Bot Logger Initialized")
LOGGER.info(f"📝 Log file: {log_file}")
LOGGER.info("="*70)