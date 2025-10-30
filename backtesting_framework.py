# backtesting_framework.py - PRODUCTION-READY BACKTESTING
"""
Complete backtesting framework with:
- Historical data replay
- Performance analytics
- Walk-forward analysis
- Monte Carlo simulation
- Strategy comparison
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass, asdict
import logging

from config import Config
from ppo_agent import PPOAgent
from data_pipeline import ExchangeDataSource
from exchange_manager import RiskManager, ExchangeManager
from trading_env import TradingBotEnv
from utils import calculate_metrics, sharpe_ratio, max_drawdown
from plot_utils import (plot_equity_curve, plot_drawdown, 
                       plot_returns_distribution, create_performance_dashboard)

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    
    # Performance metrics
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    # Equity curve
    equity_curve: List[float]
    returns: List[float]
    
    # Trades
    trades: List[Dict]
    
    # Timing
    start_date: str
    end_date: str
    duration_days: int
    
    # Configuration
    config: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"💾 Backtest results saved: {path}")


class BacktestEngine:
    """
    Complete backtesting engine with:
    - Historical data replay
    - Multiple evaluation periods
    - Performance analytics
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.results: List[BacktestResult] = []
    
    def run_backtest(self, agent: PPOAgent, start_date: Optional[str] = None,
                    end_date: Optional[str] = None, 
                    deterministic: bool = True) -> BacktestResult:
        """
        Run single backtest.
        
        Args:
            agent: trained agent
            start_date: start date (None = from beginning)
            end_date: end date (None = to end)
            deterministic: use greedy actions
        
        Returns:
            BacktestResult
        """
        logger.info(f"🔄 Running backtest: {start_date} to {end_date}")
        
        # Initialize environment
        data_source = ExchangeDataSource(self.config)
        data_source.load_initial_data()
        
        # Filter data by date if specified
        if start_date or end_date:
            data_source.data_store = self._filter_by_date(
                data_source.data_store, start_date, end_date
            )
        
        rm = RiskManager(self.config)
        em = ExchangeManager(rm, self.config)
        env = TradingBotEnv(data_source, em, self.config)
        
        # Run episode
        obs, _ = env.reset()
        agent.reset_recurrent_state()
        done = False
        
        equity_curve = [self.config.initial_balance]
        trades = []
        
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            equity_curve.append(info['net_worth'])
            
            # Record trades
            if len(env.trade_log) > len(trades):
                trades = env.trade_log.copy()
            
            obs = next_obs
        
        # Calculate metrics
        equity_array = np.array(equity_curve)
        returns_array = np.diff(equity_array) / equity_array[:-1]
        
        metrics = calculate_metrics(equity_array, trades)
        
        # Get dates
        start = data_source.data_store.index[0] if hasattr(data_source.data_store.index, 'min') else start_date or "N/A"
        end = data_source.data_store.index[-1] if hasattr(data_source.data_store.index, 'max') else end_date or "N/A"
        
        if hasattr(data_source.data_store.index, 'min'):
            duration = (data_source.data_store.index[-1] - data_source.data_store.index[0]).days
        else:
            duration = len(data_source.data_store)
        
        result = BacktestResult(
            total_return=metrics.get('total_return', 0.0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
            sortino_ratio=metrics.get('sortino_ratio', 0.0),
            calmar_ratio=metrics.get('calmar_ratio', 0.0),
            max_drawdown=metrics.get('max_drawdown', 0.0),
            total_trades=metrics.get('total_trades', 0),
            winning_trades=metrics.get('winning_trades', 0),
            losing_trades=metrics.get('losing_trades', 0),
            win_rate=metrics.get('win_rate', 0.0),
            profit_factor=metrics.get('profit_factor', 0.0),
            avg_win=metrics.get('avg_win', 0.0),
            avg_loss=metrics.get('avg_loss', 0.0),
            equity_curve=equity_curve,
            returns=returns_array.tolist(),
            trades=trades,
            start_date=str(start),
            end_date=str(end),
            duration_days=duration,
            config=self.config.to_dict()
        )
        
        self.results.append(result)
        
        logger.info(f"✅ Backtest complete: Return={result.total_return:.2%}, "
                   f"Sharpe={result.sharpe_ratio:.2f}, WR={result.win_rate:.2%}")
        
        return result
    
    def walk_forward_analysis(self, agent: PPOAgent, 
                             train_periods: int = 6,
                             test_periods: int = 2,
                             step_periods: int = 1) -> List[BacktestResult]:
        """
        Walk-forward analysis.
        
        Args:
            agent: agent to test
            train_periods: number of periods for training window
            test_periods: number of periods for testing window
            step_periods: step size for rolling window
        
        Returns:
            List of backtest results
        """
        logger.info(f"📊 Starting walk-forward analysis: "
                   f"train={train_periods}, test={test_periods}, step={step_periods}")
        
        # Load full dataset
        data_source = ExchangeDataSource(self.config)
        data_source.load_initial_data()
        
        data = data_source.data_store
        total_periods = len(data)
        
        results = []
        
        # Rolling window
        start_idx = 0
        while start_idx + train_periods + test_periods <= total_periods:
            train_end = start_idx + train_periods
            test_end = train_end + test_periods
            
            # Get date ranges
            if hasattr(data.index, 'min'):
                train_start_date = str(data.index[start_idx])
                train_end_date = str(data.index[train_end - 1])
                test_start_date = str(data.index[train_end])
                test_end_date = str(data.index[test_end - 1])
            else:
                train_start_date = f"idx_{start_idx}"
                train_end_date = f"idx_{train_end - 1}"
                test_start_date = f"idx_{train_end}"
                test_end_date = f"idx_{test_end - 1}"
            
            logger.info(f"Testing period: {test_start_date} to {test_end_date}")
            
            # Run backtest on test period
            result = self.run_backtest(
                agent,
                start_date=test_start_date,
                end_date=test_end_date,
                deterministic=True
            )
            
            results.append(result)
            
            # Step forward
            start_idx += step_periods
        
        logger.info(f"✅ Walk-forward analysis complete: {len(results)} periods tested")
        
        return results
    
    def monte_carlo_simulation(self, returns: np.ndarray, 
                               initial_capital: float = 100000,
                               num_simulations: int = 1000) -> Dict:
        """
        Monte Carlo simulation of returns.
        
        Args:
            returns: historical returns
            initial_capital: starting capital
            num_simulations: number of simulations
        
        Returns:
            Dict with simulation results
        """
        logger.info(f"🎲 Running Monte Carlo simulation: {num_simulations} runs")
        
        num_periods = len(returns)
        
        # Bootstrap returns
        simulated_curves = []
        final_values = []
        max_dds = []
        
        for _ in range(num_simulations):
            # Resample returns with replacement
            sim_returns = np.random.choice(returns, size=num_periods, replace=True)
            
            # Generate equity curve
            equity = initial_capital * np.cumprod(1 + sim_returns)
            equity = np.insert(equity, 0, initial_capital)
            
            simulated_curves.append(equity)
            final_values.append(equity[-1])
            max_dds.append(max_drawdown(equity))
        
        # Calculate statistics
        final_values = np.array(final_values)
        max_dds = np.array(max_dds)
        
        results = {
            'mean_final_value': float(np.mean(final_values)),
            'median_final_value': float(np.median(final_values)),
            'std_final_value': float(np.std(final_values)),
            'percentile_5': float(np.percentile(final_values, 5)),
            'percentile_95': float(np.percentile(final_values, 95)),
            'mean_max_dd': float(np.mean(max_dds)),
            'worst_max_dd': float(np.max(max_dds)),
            'prob_profit': float(np.mean(final_values > initial_capital)),
            'simulated_curves': [curve.tolist() for curve in simulated_curves[:100]]  # Store first 100
        }
        
        logger.info(f"✅ Monte Carlo complete: "
                   f"Mean final={results['mean_final_value']:,.2f}, "
                   f"P(profit)={results['prob_profit']:.2%}")
        
        return results
    
    def compare_strategies(self, agents: Dict[str, PPOAgent],
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            agents: dict of {name: agent}
            start_date: start date
            end_date: end date
        
        Returns:
            DataFrame with comparison
        """
        logger.info(f"📊 Comparing {len(agents)} strategies")
        
        comparison = []
        
        for name, agent in agents.items():
            result = self.run_backtest(agent, start_date, end_date)
            
            comparison.append({
                'Strategy': name,
                'Return': result.total_return,
                'Sharpe': result.sharpe_ratio,
                'Sortino': result.sortino_ratio,
                'Max DD': result.max_drawdown,
                'Win Rate': result.win_rate,
'Profit Factor': result.profit_factor,
                'Total Trades': result.total_trades
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Sharpe', ascending=False)
        
        logger.info(f"✅ Strategy comparison complete")
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        return df
    
    def _filter_by_date(self, df: pd.DataFrame, 
                       start_date: Optional[str], 
                       end_date: Optional[str]) -> pd.DataFrame:
        """Filter dataframe by date range."""
        if not hasattr(df.index, 'min'):
            return df
        
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        return df
    
    def generate_report(self, result: BacktestResult, save_dir: str = "runs/backtest"):
        """
        Generate comprehensive backtest report.
        
        Args:
            result: backtest result
            save_dir: directory to save report
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save JSON
        result.save(os.path.join(save_dir, "backtest_result.json"))
        
        # Generate plots
        equity_array = np.array(result.equity_curve)
        
        plot_equity_curve(
            equity_array, result.trades,
            title="Backtest Equity Curve",
            save_path=os.path.join(save_dir, "equity_curve.html")
        )
        
        plot_drawdown(
            equity_array,
            title="Backtest Drawdown",
            save_path=os.path.join(save_dir, "drawdown.html")
        )
        
        if len(result.returns) > 0:
            plot_returns_distribution(
                np.array(result.returns),
                title="Backtest Returns Distribution",
                save_path=os.path.join(save_dir, "returns_dist.html")
            )
        
        # Generate HTML report
        self._generate_html_report(result, save_dir)
        
        logger.info(f"✅ Backtest report generated: {save_dir}")
    
    def _generate_html_report(self, result: BacktestResult, save_dir: str):
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #e0e0e0; }}
                h1 {{ color: #4CAF50; }}
                h2 {{ color: #2196F3; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #555; padding: 12px; text-align: left; }}
                th {{ background-color: #2196F3; color: white; }}
                tr:nth-child(even) {{ background-color: #2a2a2a; }}
                .metric {{ font-weight: bold; color: #4CAF50; }}
                .negative {{ color: #f44336; }}
                .timestamp {{ color: #888; font-size: 0.9em; }}
                iframe {{ width: 100%; height: 600px; border: 1px solid #555; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>📊 Backtest Report</h1>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>📈 Performance Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Period</td><td>{result.start_date} to {result.end_date}</td></tr>
                <tr><td>Duration</td><td>{result.duration_days} days</td></tr>
                <tr><td>Total Return</td><td class="{'metric' if result.total_return > 0 else 'negative'}">{result.total_return:.2%}</td></tr>
                <tr><td>Sharpe Ratio</td><td class="metric">{result.sharpe_ratio:.2f}</td></tr>
                <tr><td>Sortino Ratio</td><td class="metric">{result.sortino_ratio:.2f}</td></tr>
                <tr><td>Calmar Ratio</td><td class="metric">{result.calmar_ratio:.2f}</td></tr>
                <tr><td>Max Drawdown</td><td class="negative">{result.max_drawdown:.2%}</td></tr>
            </table>
            
            <h2>💼 Trading Statistics</h2>
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{result.total_trades}</td></tr>
                <tr><td>Winning Trades</td><td class="metric">{result.winning_trades}</td></tr>
                <tr><td>Losing Trades</td><td class="negative">{result.losing_trades}</td></tr>
                <tr><td>Win Rate</td><td class="{'metric' if result.win_rate > 0.5 else 'negative'}">{result.win_rate:.2%}</td></tr>
                <tr><td>Profit Factor</td><td class="metric">{result.profit_factor:.2f}</td></tr>
                <tr><td>Average Win</td><td class="metric">${result.avg_win:.2f}</td></tr>
                <tr><td>Average Loss</td><td class="negative">${result.avg_loss:.2f}</td></tr>
                <tr><td>Final Equity</td><td class="metric">${result.equity_curve[-1]:,.2f}</td></tr>
            </table>
            
            <h2>📊 Equity Curve</h2>
            <iframe src="equity_curve.html"></iframe>
            
            <h2>📉 Drawdown Analysis</h2>
            <iframe src="drawdown.html"></iframe>
            
            <h2>📈 Returns Distribution</h2>
            <iframe src="returns_dist.html"></iframe>
        </body>
        </html>
        """
        
        with open(os.path.join(save_dir, "report.html"), 'w') as f:
            f.write(html)