"""
Backtesting Engine for AlgoGPT + Kronos
========================================
Walk-forward simulation with realistic execution.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd

from .features import FeatureEngineer
from .signals import SignalGenerator, SignalConfig
from ..model import KronosPredictor

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100_000.0
    commission: float = 0.001       # 0.1% per trade
    slippage: float = 0.0005        # 0.05% slippage
    max_position_pct: float = 0.95  # Max 95% of capital in one position
    lookback: int = 200
    pred_len: int = 5
    step: int = 1
    kronos_weight: float = 0.65
    buy_threshold: float = 0.20
    sell_threshold: float = -0.20
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    use_macd_filter: bool = True
    min_confidence: float = 0.5
    max_risk_pct: float = 0.02


@dataclass
class BacktestResults:
    """Container for backtest performance metrics."""
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    signals: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        no_trades = m.get("executed_orders", 0) == 0
        lines = [
            "=" * 55,
            "  AlgoGPT + Kronos  –  Backtest Summary (Indian Derivatives)",
            "=" * 55,
            f"  Total Return     : {m.get('total_return', 0):+.2%}" + ("  (flat: no orders executed)" if no_trades else ""),
            f"  Benchmark Return : {m.get('benchmark_return', 0):+.2%}",
            f"  Alpha            : {m.get('alpha', 0):+.2%}",
            f"  Signals          : buy={m.get('buy_signals', 0)}  sell={m.get('sell_signals', 0)}  hold={m.get('hold_signals', 0)}",
            f"  Orders Executed  : {m.get('executed_orders', 0)}",
        ]
        if no_trades:
            lines.append("  Trade Metrics    : N/A (no completed position)")
        else:
            lines.extend([
                f"  Sharpe Ratio     : {m.get('sharpe', 0):.3f}",
                f"  Max Drawdown     : {m.get('max_drawdown', 0):.2%}",
                f"  Win Rate         : {m.get('win_rate', 0):.1%}",
                f"  Profit Factor    : {m.get('profit_factor', 0):.2f}",
                f"  Closed Trades    : {m.get('num_trades', 0)}",
            ])
        lines.extend([
            f"  Kronos Available : {m.get('kronos_available', 'unknown')}",
            "=" * 55,
        ])
        return "\n".join(lines)

    def plot(self, title: str = "AlgoGPT + Kronos Backtest") -> None:
        """Plot equity curve vs benchmark. Requires matplotlib."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker

            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            fig.suptitle(title, fontsize=14, fontweight="bold")

            # Equity curves
            ax1 = axes[0]
            self.equity_curve.plot(ax=ax1, label="AlgoGPT + Kronos", color="#4CAF50", linewidth=2)
            self.benchmark_curve.plot(ax=ax1, label="Buy & Hold", color="#2196F3",
                                      linewidth=1.5, linestyle="--")
            ax1.set_ylabel("Portfolio Value ($)")
            ax1.legend(loc="upper left")
            ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax1.grid(True, alpha=0.3)

            # Drawdown
            ax2 = axes[1]
            peak = self.equity_curve.cummax()
            drawdown = (self.equity_curve - peak) / peak
            drawdown.plot(ax=ax2, color="#F44336", linewidth=1.5)
            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="#F44336")
            ax2.set_ylabel("Drawdown")
            ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax2.grid(True, alpha=0.3)

            # Signals
            ax3 = axes[2]
            if not self.signals.empty and "strength" in self.signals.columns:
                self.signals["strength"].plot(ax=ax3, color="#9C27B0", linewidth=1.0, label="Signal Strength")
                ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--")
                ax3.set_ylabel("Signal Strength")
                ax3.legend(loc="upper left")
                ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("matplotlib not installed – cannot plot results.")


class KronosBacktester:
    """
    Walk-forward backtester for AlgoGPT + Kronos on Indian derivatives.

    Execution model:
    - Signal generated at bar i using lookback bars [i-lookback, i)
    - Trade executed at bar i+1 open price (realistic next-bar execution)
    - Position marked to market at each bar close
    """

    def __init__(
        self,
        predictor: KronosPredictor,
        config: Optional[BacktestConfig] = None,
    ):
        self.predictor = predictor
        self.config = config or BacktestConfig()

    def run(
        self,
        df: pd.DataFrame,
        timestamps: pd.Series,
        verbose: bool = True,
    ) -> BacktestResults:
        """
        Execute walk-forward backtest.

        Parameters
        ----------
        df : pd.DataFrame
            Full OHLCV DataFrame (lower-case columns: open, high, low, close, volume, amount)
        timestamps : pd.Series
            DatetimeSeries aligned 1-to-1 with df rows
        verbose : bool
            Print progress

        Returns
        -------
        BacktestResults
        """
        fe = FeatureEngineer()
        sg = SignalGenerator(SignalConfig(
            kronos_weight=self.config.kronos_weight,
            buy_threshold=self.config.buy_threshold,
            sell_threshold=self.config.sell_threshold,
            rsi_overbought=self.config.rsi_overbought,
            rsi_oversold=self.config.rsi_oversold,
            use_macd_filter=self.config.use_macd_filter,
        ))

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = fe.compute(df)

        n = len(df)
        required = self.config.lookback + self.config.pred_len + 1
        if n < required:
            raise ValueError(f"DataFrame has {n} rows but needs at least {required}")

        # State
        capital = self.config.initial_capital
        position = 0
        entry_price = 0.0

        equity_records = []
        signal_records = []
        trade_records = []

        # Benchmark: buy-and-hold from first tradable bar
        bm_entry_idx = self.config.lookback
        bm_entry_price = float(df["open"].iloc[bm_entry_idx])
        bm_shares = (self.config.initial_capital * self.config.max_position_pct) / bm_entry_price

        # Trading indices
        indices = list(range(self.config.lookback, n - self.config.pred_len, self.config.step))
        total = len(indices)

        for rank, i in enumerate(indices):
            if verbose and rank % 20 == 0:
                logger.info("Backtesting ... step %d / %d", rank, total)

            # Historical window for Kronos
            x_df = df.iloc[i - self.config.lookback : i][["open", "high", "low", "close", "volume", "amount"]]
            x_timestamp = timestamps.iloc[i - self.config.lookback : i]
            y_timestamp = timestamps.iloc[i : i + self.config.pred_len]
            current_row = df.iloc[i - 1]  # Most recent completed bar

            # Kronos forecast
            try:
                pred_df = self.predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=self.config.pred_len,
                    verbose=False,
                )
            except Exception as exc:
                logger.warning("Forecast failed at bar %d: %s", i, exc)
                # Record equity without trading
                current_price = float(df["close"].iloc[i])
                portfolio_val = capital + position * current_price
                equity_records.append((timestamps.iloc[i], portfolio_val))
                continue

            # Generate signal
            signal = sg.generate(current_row, pred_df)

            # Execution price = next bar open with slippage
            exec_price = float(df["open"].iloc[i]) * (1 + self.config.slippage * signal.direction)

            signal_records.append({
                "timestamp": timestamps.iloc[i],
                "direction": signal.direction,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "kronos_dir": signal.kronos_direction,
                "ta_dir": signal.ta_direction,
            })

            # Trade execution
            current_close = float(df["close"].iloc[i])

            if signal.direction == 1 and position == 0:
                # BUY
                affordable = capital * self.config.max_position_pct
                shares = int(affordable / exec_price)
                if shares > 0:
                    cost = shares * exec_price * (1 + self.config.commission)
                    capital -= cost
                    position = shares
                    entry_price = exec_price
                    trade_records.append({
                        "timestamp": timestamps.iloc[i],
                        "action": "BUY",
                        "price": exec_price,
                        "shares": shares,
                        "cost": cost,
                    })

            elif signal.direction == -1 and position > 0:
                # SELL
                proceeds = position * exec_price * (1 - self.config.commission)
                pnl = proceeds - position * entry_price
                capital += proceeds
                trade_records.append({
                    "timestamp": timestamps.iloc[i],
                    "action": "SELL",
                    "price": exec_price,
                    "shares": position,
                    "proceeds": proceeds,
                    "pnl": pnl,
                })
                position = 0
                entry_price = 0.0

            # Mark-to-market
            portfolio_val = capital + position * current_close
            equity_records.append((timestamps.iloc[i], portfolio_val))

        # Close any open position at the final available close so realised PnL,
        # terminal equity, and trade metrics describe the same backtest window.
        if position > 0:
            last_price = float(df["close"].iloc[-1])
            proceeds = position * last_price * (1 - self.config.commission)
            pnl = proceeds - position * entry_price
            capital += proceeds
            trade_records.append({
                "timestamp": timestamps.iloc[-1],
                "action": "SELL",
                "price": last_price,
                "shares": position,
                "proceeds": proceeds,
                "pnl": pnl,
                "reason": "final_close",
            })
            position = 0
            entry_price = 0.0
            equity_records.append((timestamps.iloc[-1], capital))

        # Build results
        equity_idx, equity_vals = zip(*equity_records) if equity_records else ([], [])
        equity_curve = pd.Series(equity_vals, index=equity_idx, name="portfolio")

        # Benchmark equity curve
        bm_vals = (
            bm_shares * df["close"].loc[df.index.isin(equity_idx)]
            + (self.config.initial_capital - bm_shares * bm_entry_price)
        )
        bm_vals.index = equity_curve.index
        bm_vals.name = "benchmark"

        signals_df = pd.DataFrame(signal_records).set_index("timestamp") if signal_records else pd.DataFrame()
        trades_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()

        metrics = self._compute_metrics(equity_curve, bm_vals, trades_df)
        direction_counts = signals_df["direction"].value_counts() if not signals_df.empty else pd.Series(dtype=int)
        metrics.update({
            "buy_signals": int(direction_counts.get(1, 0)),
            "sell_signals": int(direction_counts.get(-1, 0)),
            "hold_signals": int(direction_counts.get(0, 0)),
            "executed_orders": len(trades_df),
            "kronos_available": self.predictor.is_kronos_loaded,
        })

        return BacktestResults(
            equity_curve=equity_curve,
            benchmark_curve=bm_vals,
            signals=signals_df,
            trades=trades_df,
            metrics=metrics,
        )

    @staticmethod
    def _compute_metrics(
        equity: pd.Series,
        benchmark: pd.Series,
        trades: pd.DataFrame,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> dict:
        if equity.empty:
            return {}

        returns = equity.pct_change().dropna()
        bm_returns = benchmark.pct_change().dropna()

        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        bm_return = (benchmark.iloc[-1] / benchmark.iloc[0]) - 1

        # Sharpe
        excess = returns - risk_free_rate / periods_per_year
        sharpe = (
            (excess.mean() / (excess.std() + 1e-9)) * np.sqrt(periods_per_year)
            if len(excess) > 1 else 0.0
        )

        # Max drawdown
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = float(drawdown.min())

        # Win rate & profit factor
        num_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        if not trades.empty and "pnl" in trades.columns:
            pnls = trades["pnl"].dropna()
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            num_trades = len(pnls)
            win_rate = len(wins) / max(num_trades, 1)
            profit_factor = (
                wins.sum() / (-losses.sum() + 1e-9)
                if len(losses) > 0 else float("inf")
            )

        return {
            "total_return": total_return,
            "benchmark_return": bm_return,
            "alpha": total_return - bm_return,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "num_trades": num_trades,
        }


def run_backtest(
    predictor: KronosPredictor,
    df: pd.DataFrame,
    timestamps: pd.Series,
    config: Optional[BacktestConfig] = None,
    verbose: bool = True,
) -> BacktestResults:
    """Convenience function to run a backtest."""
    bt = KronosBacktester(predictor, config)
    return bt.run(df, timestamps, verbose)