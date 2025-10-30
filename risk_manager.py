import numpy as np, time
# from config import Config # Assuming Config is available via import

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.max_drawdown = config.max_drawdown_pct
        self.equity = config.initial_balance  # use config.initial_balance
        self.peak_equity = config.initial_balance
        self.halted = False


    def update_equity(self, equity):
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - self.equity) / (self.peak_equity + 1e-9)
        if dd >= self.max_drawdown:
            self.halted = True

    def allow_order(self, size_fraction):
        if self.halted:
            return False, "Halted due to drawdown"
        if abs(size_fraction) > self.config.max_position_size:
            return False, "Order exceeds max position size"
        return True, ""