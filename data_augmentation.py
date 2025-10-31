# data_augmentation.py - NEW FILE FOR DATA DIVERSITY
"""
Generate diverse market scenarios for training.

Solves the problem of training only on low-volatility data
by creating:
- Bull runs
- Bear markets
- High volatility
- Range-bound markets
- Flash crashes
- Trend reversals
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class MarketScenarioGenerator:
    """
    Generate diverse training scenarios from base market data.
    
    Usage:
        generator = MarketScenarioGenerator(historical_ohlcv_df)
        scenarios = generator.generate_all_scenarios()
        
        for name, data in scenarios:
            env.load_data(data)
            # Train for N episodes
    """
    
    def __init__(self, base_df: pd.DataFrame):
        """
        Initialize with base OHLCV data.
        
        Args:
            base_df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.base = base_df.copy()
        
        # Validate columns
        required_cols = ['open', 'high', 'low', 'close']
        missing = [col for col in required_cols if col not in self.base.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        logger.info(f"📊 MarketScenarioGenerator initialized with {len(self.base)} candles")
    
    def generate_all_scenarios(self) -> List[Tuple[str, pd.DataFrame]]:
        """
        Generate all market scenarios.
        
        Returns:
            List of (scenario_name, dataframe) tuples
        """
        scenarios = []
        
        # 1. Original data
        scenarios.append(('original', self.base.copy()))
        logger.info("  ✓ Original scenario")
        
        # 2. Bull run
        bull = self._create_bull_run()
        scenarios.append(('bull_run', bull))
        logger.info("  ✓ Bull run scenario (+30%)")
        
        # 3. Bear market
        bear = self._create_bear_market()
        scenarios.append(('bear_market', bear))
        logger.info("  ✓ Bear market scenario (-25%)")
        
        # 4. High volatility
        volatile = self._create_high_volatility()
        scenarios.append(('high_volatility', volatile))
        logger.info("  ✓ High volatility scenario (±5%)")
        
        # 5. Range bound
        ranging = self._create_range_bound()
        scenarios.append(('range_bound', ranging))
        logger.info("  ✓ Range-bound scenario")
        
        # 6. Flash crash
        crash = self._create_flash_crash()
        scenarios.append(('flash_crash', crash))
        logger.info("  ✓ Flash crash scenario (-15% → recovery)")
        
        # 7. Trend reversal
        reversal = self._create_trend_reversal()
        scenarios.append(('trend_reversal', reversal))
        logger.info("  ✓ Trend reversal scenario")
        
        # 8. Whipsaw
        whipsaw = self._create_whipsaw()
        scenarios.append(('whipsaw', whipsaw))
        logger.info("  ✓ Whipsaw scenario (false breakouts)")
        
        logger.info(f"📊 Generated {len(scenarios)} market scenarios")
        
        return scenarios
    
    def _create_bull_run(self, gain_pct: float = 0.30) -> pd.DataFrame:
        """Create steady bull run scenario."""
        df = self.base.copy()
        multiplier = np.linspace(1.0, 1.0 + gain_pct, len(df))
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * multiplier
        
        return df
    
    def _create_bear_market(self, loss_pct: float = 0.25) -> pd.DataFrame:
        """Create steady bear market scenario."""
        df = self.base.copy()
        multiplier = np.linspace(1.0, 1.0 - loss_pct, len(df))
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * multiplier
        
        return df
    
    def _create_high_volatility(self, noise_std: float = 0.025) -> pd.DataFrame:
        """Create high volatility with random noise."""
        df = self.base.copy()
        
        # Set seed for reproducibility
        np.random.seed(42)
        noise = np.random.normal(0, noise_std, len(df))
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * (1 + noise)
        
        # Ensure OHLC relationship is maintained
        df = self._fix_ohlc_relationship(df)
        
        return df
    
    def _create_range_bound(self, range_pct: float = 0.03) -> pd.DataFrame:
        """Create oscillating range-bound market."""
        df = self.base.copy()
        
        mean_price = df['close'].mean()
        
        # Create sine wave oscillation
        periods = 6  # Number of complete cycles
        sine_wave = range_pct * mean_price * np.sin(
            np.linspace(0, periods * 2 * np.pi, len(df))
        )
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = mean_price + sine_wave
        
        return df
    
    def _create_flash_crash(self, crash_pct: float = 0.15) -> pd.DataFrame:
        """Create sudden crash with recovery."""
        df = self.base.copy()
        
        # Crash happens at 1/3 point
        crash_point = len(df) // 3
        # Recovery happens at 1/4 later
        recovery_point = crash_point + len(df) // 4
        
        multiplier = np.ones(len(df))
        
        # Normal before crash
        multiplier[:crash_point] = 1.0
        
        # Sudden crash
        multiplier[crash_point:recovery_point] = np.linspace(
            1.0, 1.0 - crash_pct, recovery_point - crash_point
        )
        
        # Gradual recovery (but not full)
        multiplier[recovery_point:] = np.linspace(
            1.0 - crash_pct, 1.0 - (crash_pct * 0.3), len(df) - recovery_point
        )
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * multiplier
        
        return df
    
    def _create_trend_reversal(self) -> pd.DataFrame:
        """Create uptrend → downtrend reversal."""
        df = self.base.copy()
        
        mid = len(df) // 2
        
        # First half: uptrend
        multiplier_up = np.linspace(1.0, 1.15, mid)
        # Second half: downtrend
        multiplier_down = np.linspace(1.0, 0.90, len(df) - mid)
        
        multiplier = np.concatenate([multiplier_up, multiplier_down])
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * multiplier
        
        return df
    
    def _create_whipsaw(self, num_whipsaws: int = 5) -> pd.DataFrame:
        """Create false breakout pattern (whipsaw)."""
        df = self.base.copy()
        
        mean_price = df['close'].mean()
        
        # Create sharp oscillations
        segment_length = len(df) // num_whipsaws
        multiplier = np.ones(len(df))
        
        for i in range(num_whipsaws):
            start = i * segment_length
            end = min((i + 1) * segment_length, len(df))
            
            # Alternate between spikes up and down
            if i % 2 == 0:
                # Spike up then revert
                spike = np.concatenate([
                    np.linspace(1.0, 1.05, (end - start) // 2),
                    np.linspace(1.05, 1.0, (end - start) // 2)
                ])
            else:
                # Spike down then revert
                spike = np.concatenate([
                    np.linspace(1.0, 0.95, (end - start) // 2),
                    np.linspace(0.95, 1.0, (end - start) // 2)
                ])
            
            multiplier[start:end] = spike[:end - start]
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * multiplier
        
        return df
    
    def _fix_ohlc_relationship(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLC maintains correct relationships (H >= O,C,L and L <= O,C,H)."""
        # High should be max of O, H, C
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        
        # Low should be min of O, L, C
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def create_custom_scenario(
        self,
        name: str,
        trend: str = 'neutral',
        volatility: str = 'normal',
        events: List[dict] = None
    ) -> Tuple[str, pd.DataFrame]:
        """
        Create custom scenario.
        
        Args:
            name: Scenario name
            trend: 'up', 'down', 'neutral'
            volatility: 'low', 'normal', 'high'
            events: List of {'type': 'crash'/'pump', 'start': 0.5, 'magnitude': 0.1}
        
        Returns:
            (name, dataframe) tuple
        """
        df = self.base.copy()
        
        # Apply trend
        if trend == 'up':
            multiplier = np.linspace(1.0, 1.20, len(df))
        elif trend == 'down':
            multiplier = np.linspace(1.0, 0.85, len(df))
        else:
            multiplier = np.ones(len(df))
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * multiplier
        
        # Apply volatility
        if volatility == 'high':
            noise = np.random.normal(0, 0.03, len(df))
        elif volatility == 'low':
            noise = np.random.normal(0, 0.005, len(df))
        else:
            noise = np.random.normal(0, 0.015, len(df))
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col] * (1 + noise)
        
        # Apply events
        if events:
            for event in events:
                event_type = event.get('type', 'crash')
                start_pct = event.get('start', 0.5)
                magnitude = event.get('magnitude', 0.1)
                
                start_idx = int(len(df) * start_pct)
                duration = min(100, len(df) - start_idx)
                
                event_mult = np.ones(len(df))
                if event_type == 'crash':
                    event_mult[start_idx:start_idx + duration] = np.linspace(
                        1.0, 1.0 - magnitude, duration
                    )
                elif event_type == 'pump':
                    event_mult[start_idx:start_idx + duration] = np.linspace(
                        1.0, 1.0 + magnitude, duration
                    )
                
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = df[col] * event_mult
        
        df = self._fix_ohlc_relationship(df)
        
        return (name, df)
    
    def get_scenario_statistics(self, scenarios: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        """
        Calculate statistics for all scenarios.
        
        Returns:
            DataFrame with scenario statistics
        """
        stats = []
        
        for name, df in scenarios:
            returns = df['close'].pct_change().dropna()
            
            stat = {
                'scenario': name,
                'length': len(df),
                'total_return': (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100,
                'volatility': returns.std() * np.sqrt(len(df)) * 100,
                'max_drawdown': self._calculate_max_drawdown(df['close']) * 100,
                'mean_price': df['close'].mean(),
                'min_price': df['close'].min(),
                'max_price': df['close'].max(),
            }
            stats.append(stat)
        
        return pd.DataFrame(stats)
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        return drawdown.min()


# ========== USAGE EXAMPLE ==========
if __name__ == '__main__':
    # Example usage
    print("📊 Market Scenario Generator")
    print("=" * 70)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    base_price = 40000
    noise = np.random.normal(0, 100, 1000)
    
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': base_price + noise,
        'high': base_price + noise + 50,
        'low': base_price + noise - 50,
        'close': base_price + noise + np.random.normal(0, 20, 1000),
        'volume': np.random.uniform(1, 100, 1000)
    })
    
    # Generate scenarios
    generator = MarketScenarioGenerator(sample_df)
    scenarios = generator.generate_all_scenarios()
    
    # Print statistics
    stats = generator.get_scenario_statistics(scenarios)
    print("\n📈 Scenario Statistics:")
    print(stats.to_string(index=False))
    
    print("\n✅ All scenarios generated successfully!")
    print(f"   Use these to train your agent on diverse market conditions")