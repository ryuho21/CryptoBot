# ab_testing_framework.py - A/B TESTING FRAMEWORK
"""
Complete A/B testing framework for:
- Strategy comparison
- Hyperparameter optimization
- Statistical significance testing
- Multi-armed bandit allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import os
from scipy import stats
import logging

from config import Config
from ppo_agent import PPOAgent
from backtesting_framework import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class VariantResult:
    """Results for a single variant."""
    variant_name: str
    config: Dict
    num_trials: int
    mean_return: float
    std_return: float
    mean_sharpe: float
    std_sharpe: float
    mean_win_rate: float
    total_trades: int
    equity_curves: List[List[float]]
    backtest_results: List[BacktestResult]
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        # Convert BacktestResult objects to dicts
        result['backtest_results'] = [r.to_dict() for r in self.backtest_results]
        return result


class ABTestFramework:
    """
    A/B testing framework for strategy comparison.
    
    Features:
    - Multiple variant testing
    - Statistical significance
    - Bayesian optimization
    - Multi-armed bandit
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.variants: Dict[str, VariantResult] = {}
        self.backtest_engine = BacktestEngine(config)
    
    def add_variant(self, name: str, agent: PPOAgent, 
                   config: Optional[Config] = None) -> str:
        """
        Add a variant to test.
        
        Args:
            name: variant identifier
            agent: trained agent
            config: optional config override
        
        Returns:
            variant_id: unique identifier
        """
        variant_id = f"{name}_{len(self.variants)}"
        
        logger.info(f"➕ Added variant: {variant_id}")
        
        return variant_id
    
    def run_ab_test(self, agents: Dict[str, PPOAgent],
                   num_trials: int = 10,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> Dict[str, VariantResult]:
        """
        Run A/B test comparing multiple agents.
        
        Args:
            agents: dict of {variant_name: agent}
            num_trials: number of backtest trials per variant
            start_date: start date for backtest
            end_date: end date for backtest
        
        Returns:
            dict of {variant_name: VariantResult}
        """
        logger.info(f"🧪 Starting A/B test with {len(agents)} variants, {num_trials} trials each")
        
        results = {}
        
        for variant_name, agent in agents.items():
            logger.info(f"Testing variant: {variant_name}")
            
            variant_results = []
            equity_curves = []
            
            for trial in range(num_trials):
                logger.info(f"  Trial {trial + 1}/{num_trials}")
                
                # Run backtest
                result = self.backtest_engine.run_backtest(
                    agent,
                    start_date=start_date,
                    end_date=end_date,
                    deterministic=True
                )
                
                variant_results.append(result)
                equity_curves.append(result.equity_curve)
            
            # Aggregate statistics
            returns = [r.total_return for r in variant_results]
            sharpes = [r.sharpe_ratio for r in variant_results]
            win_rates = [r.win_rate for r in variant_results]
            total_trades = sum(r.total_trades for r in variant_results)
            
            variant_summary = VariantResult(
                variant_name=variant_name,
                config=agent.config.to_dict() if hasattr(agent, 'config') else {},
                num_trials=num_trials,
                mean_return=float(np.mean(returns)),
                std_return=float(np.std(returns)),
                mean_sharpe=float(np.mean(sharpes)),
                std_sharpe=float(np.std(sharpes)),
                mean_win_rate=float(np.mean(win_rates)),
                total_trades=total_trades,
                equity_curves=equity_curves,
                backtest_results=variant_results
            )
            
            results[variant_name] = variant_summary
            
            logger.info(f"  ✅ Variant {variant_name}: "
                       f"Return={variant_summary.mean_return:.2%} ± {variant_summary.std_return:.2%}, "
                       f"Sharpe={variant_summary.mean_sharpe:.2f} ± {variant_summary.std_sharpe:.2f}")
        
        self.variants = results
        
        return results
    
    def statistical_comparison(self, variant_a: str, variant_b: str) -> Dict:
        """
        Perform statistical comparison between two variants.
        
        Args:
            variant_a: first variant name
            variant_b: second variant name
        
        Returns:
            dict with statistical test results
        """
        if variant_a not in self.variants or variant_b not in self.variants:
            raise ValueError("Variants not found")
        
        result_a = self.variants[variant_a]
        result_b = self.variants[variant_b]
        
        returns_a = [r.total_return for r in result_a.backtest_results]
        returns_b = [r.total_return for r in result_b.backtest_results]
        
        # T-test
        t_stat, p_value = stats.ttest_ind(returns_a, returns_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(returns_a)**2 + np.std(returns_b)**2) / 2)
        cohens_d = (np.mean(returns_a) - np.mean(returns_b)) / pooled_std if pooled_std > 0 else 0
        
        # Confidence intervals
        ci_a = stats.t.interval(0.95, len(returns_a)-1, 
                                loc=np.mean(returns_a), 
                                scale=stats.sem(returns_a))
        ci_b = stats.t.interval(0.95, len(returns_b)-1,
                                loc=np.mean(returns_b),
                                scale=stats.sem(returns_b))
        
        comparison = {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'mean_diff': float(result_a.mean_return - result_b.mean_return),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_5pct': p_value < 0.05,
            'cohens_d': float(cohens_d),
            'confidence_interval_a': [float(ci_a[0]), float(ci_a[1])],
            'confidence_interval_b': [float(ci_b[0]), float(ci_b[1])],
            'winner': variant_a if result_a.mean_return > result_b.mean_return else variant_b
        }
        
        logger.info(f"📊 Statistical comparison:")
        logger.info(f"  {variant_a} vs {variant_b}")
        logger.info(f"  Mean difference: {comparison['mean_diff']:.2%}")
        logger.info(f"  p-value: {comparison['p_value']:.4f}")
        logger.info(f"  Significant: {comparison['significant_at_5pct']}")
        logger.info(f"  Winner: {comparison['winner']}")
        
        return comparison
    
    def rank_variants(self, metric: str = 'mean_sharpe') -> pd.DataFrame:
        """
        Rank variants by specified metric.
        
        Args:
            metric: metric to rank by
        
        Returns:
            DataFrame with rankings
        """
        if not self.variants:
            return pd.DataFrame()
        
        ranking_data = []
        for name, result in self.variants.items():
            ranking_data.append({
                'Variant': name,
                'Mean Return': result.mean_return,
                'Std Return': result.std_return,
                'Mean Sharpe': result.mean_sharpe,
                'Std Sharpe': result.std_sharpe,
                'Mean Win Rate': result.mean_win_rate,
                'Total Trades': result.total_trades,
                'Num Trials': result.num_trials
            })
        
        df = pd.DataFrame(ranking_data)
        
        # Map metric name to column
        metric_map = {
            'mean_return': 'Mean Return',
            'mean_sharpe': 'Mean Sharpe',
            'mean_win_rate': 'Mean Win Rate'
        }
        
        sort_col = metric_map.get(metric, 'Mean Sharpe')
        df = df.sort_values(sort_col, ascending=False)
        
        logger.info("\n" + "="*80)
        logger.info(f"VARIANT RANKINGS (by {sort_col})")
        logger.info("="*80)
        logger.info(df.to_string(index=False))
        logger.info("="*80 + "\n")
        
        return df
    
    def multi_armed_bandit_allocation(self, variants: List[str],
                                      epsilon: float = 0.1) -> str:
        """
        Select variant using epsilon-greedy multi-armed bandit.
        
        Args:
            variants: list of variant names
            epsilon: exploration rate
        
        Returns:
            selected_variant: name of selected variant
        """
        # Exploration
        if np.random.random() < epsilon:
            selected = np.random.choice(variants)
            logger.debug(f"🎲 Exploration: selected {selected}")
            return selected
        
        # Exploitation: select best performing
        best_variant = max(variants, key=lambda v: self.variants[v].mean_sharpe 
                          if v in self.variants else 0)
        
        logger.debug(f"🎯 Exploitation: selected {best_variant}")
        return best_variant
    
    def save_results(self, save_dir: str = "runs/ab_test"):
        """Save A/B test results."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save variant results
        for name, result in self.variants.items():
            variant_path = os.path.join(save_dir, f"variant_{name}.json")
            with open(variant_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        
        # Save rankings
        rankings = self.rank_variants()
        rankings.to_csv(os.path.join(save_dir, "rankings.csv"), index=False)
        
        # Generate comparison report
        self._generate_comparison_report(save_dir)
        
        logger.info(f"💾 A/B test results saved: {save_dir}")
    
    def _generate_comparison_report(self, save_dir: str):
        """Generate HTML comparison report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>A/B Test Comparison</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a1a; color: #e0e0e0; }
                h1 { color: #4CAF50; }
                h2 { color: #2196F3; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #555; padding: 12px; text-align: left; }
                th { background-color: #2196F3; color: white; }
                tr:nth-child(even) { background-color: #2a2a2a; }
                .best { background-color: #1b5e20; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>🧪 A/B Test Comparison Report</h1>
            <p>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            
            <h2>📊 Variant Rankings</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Variant</th>
                    <th>Mean Return</th>
                    <th>Mean Sharpe</th>
                    <th>Win Rate</th>
                    <th>Trials</th>
                </tr>
        """
        
        rankings = self.rank_variants()
        for idx, row in rankings.iterrows():
            row_class = 'best' if idx == rankings.index[0] else ''
            html += f"""
                <tr class="{row_class}">
                    <td>{idx + 1}</td>
                    <td>{row['Variant']}</td>
                    <td>{row['Mean Return']:.2%}</td>
                    <td>{row['Mean Sharpe']:.2f}</td>
                    <td>{row['Mean Win Rate']:.2%}</td>
                    <td>{row['Num Trials']}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        with open(os.path.join(save_dir, "comparison_report.html"), 'w') as f:
            f.write(html)