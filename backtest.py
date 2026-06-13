"""Event-driven backtest translating model probabilities into a strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BacktestConfig


def generate_positions(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Go long when the model predicts an upward move, flat otherwise."""
    return (probabilities > threshold).astype(np.float32)


def run_backtest(
    returns: np.ndarray,
    positions: np.ndarray,
    config: BacktestConfig,
) -> pd.DataFrame:
    """Simulate a long/flat strategy on aligned forward returns.

    `returns` are realised next-day returns aligned with each position.
    Transaction costs are charged whenever the position changes.
    """
    cost = config.cost_bps / 10_000.0
    position_changes = np.abs(np.diff(np.concatenate([[0.0], positions])))
    gross = positions * returns * config.position_size
    net = gross - position_changes * cost

    equity = config.initial_capital * np.cumprod(1.0 + net)
    benchmark = config.initial_capital * np.cumprod(1.0 + returns)

    return pd.DataFrame(
        {
            "position": positions,
            "strategy_return": net,
            "benchmark_return": returns,
            "strategy_equity": equity,
            "benchmark_equity": benchmark,
        }
    )


def performance_summary(result: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """Compute headline risk-adjusted metrics for the strategy."""
    strat = result["strategy_return"].to_numpy()
    bench = result["benchmark_return"].to_numpy()

    def annualised_sharpe(series: np.ndarray) -> float:
        if series.std() == 0:
            return 0.0
        return np.sqrt(periods_per_year) * series.mean() / series.std()

    def max_drawdown(equity: np.ndarray) -> float:
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return float(drawdown.min())

    total_return = result["strategy_equity"].iloc[-1] / result["strategy_equity"].iloc[0] - 1
    bench_return = result["benchmark_equity"].iloc[-1] / result["benchmark_equity"].iloc[0] - 1

    return {
        "strategy_total_return": float(total_return),
        "benchmark_total_return": float(bench_return),
        "strategy_sharpe": float(annualised_sharpe(strat)),
        "benchmark_sharpe": float(annualised_sharpe(bench)),
        "strategy_max_drawdown": max_drawdown(result["strategy_equity"].to_numpy()),
        "hit_rate": float((strat[result["position"].to_numpy() > 0] > 0).mean())
        if (result["position"] > 0).any()
        else 0.0,
    }
