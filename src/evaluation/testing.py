"""
Model evaluation across multiple test-environment episodes.

Calculates point estimates, confidence intervals, percentiles, and
pyfolio performance statistics over *N* rounds of interaction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm

from src.evaluation.metrics import perf_stats_series

log = logging.getLogger("deeprl_trade.evaluation")


@dataclass
class TestResult:
    """Structured container for evaluation outputs."""

    date: list = field(default_factory=list)

    # Per-round raw arrays (shape: [rounds, episode_length])
    total_reward: Optional[np.ndarray] = None
    total_profit: Optional[np.ndarray] = None
    total_profit_percentage: Optional[np.ndarray] = None
    buy_and_hold: Optional[np.ndarray] = None
    daily_price_return: Optional[np.ndarray] = None
    daily_return: Optional[np.ndarray] = None
    total_assets: Optional[np.ndarray] = None

    # Aggregated statistics (populated by post-processing)
    stats: Dict[str, Any] = field(default_factory=dict)
    perf_stat: Optional[pd.DataFrame] = None
    bnh_perf_stat: Optional[pd.Series] = None


def test_model(
    model,
    test_env,
    rounds: int = 500,
    random_start: bool = True,
) -> Dict[str, Any]:
    """
    Run *model* on *test_env* for *rounds* episodes and return statistics.

    Parameters
    ----------
    model
        A trained SB3 agent with a ``.predict()`` method.
    test_env
        A ``TradingEnv`` instance.
    rounds : int
        Number of evaluation episodes.
    random_start : bool
        Whether to randomise the start tick each episode.

    Returns
    -------
    dict
        Dictionary of aggregated results (point estimates, CIs, percentiles,
        pyfolio performance stats).
    """
    res: Dict[str, Any] = {
        "date": [],
        "total_reward": [],
        "total_profit": [],
        "total_profit_percentage": [],
        "buy_and_hold": [],
        "daily_price_return": [],
        "daily_return": [],
        "total_assets": [],
        "perf_stat": pd.DataFrame(),
    }

    for _ in tqdm(range(rounds), desc="Evaluating"):
        observation, _info = test_env.reset(
            options={"rand_trail": random_start}
        )

        while True:
            action, _states = model.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, _info = test_env.step(action)
            if terminated or truncated:
                break

        for key in test_env.history:
            res[key].append(test_env.history[key])

        # Bug #13 fix: pass returns as Series with DatetimeIndex
        episode_dates = test_env.history["date"]
        episode_returns = np.array(test_env.history["daily_return"])
        pstat = perf_stats_series(episode_returns, episode_dates)
        res["perf_stat"] = pd.concat([res["perf_stat"], pstat], axis=1)

    # Convert lists → numpy arrays
    # Episodes may have different lengths when random_start=True (same end,
    # different start). Right-align them and pad shorter ones with NaN.
    for key in list(res.keys()):
        if key not in ("date", "perf_stat"):
            episodes = res[key]
            if not episodes:
                continue
            lengths = [len(ep) for ep in episodes]
            max_len = max(lengths)
            if all(length == max_len for length in lengths):
                res[key] = np.array(episodes, dtype=np.float64)
            else:
                padded = np.full((len(episodes), max_len), np.nan)
                for i, ep in enumerate(episodes):
                    padded[i, max_len - len(ep) :] = ep
                res[key] = padded

    # Use the longest episode's dates as representative (covers full range)
    if res["date"]:
        longest_idx = max(range(len(res["date"])), key=lambda i: len(res["date"][i]))
        res["date"] = [res["date"][longest_idx]]

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------
    mean_res: Dict[str, Any] = {}

    for k, v in res.items():
        if k == "date":
            mean_res[k] = v[0]  # representative dates from first episode

        elif k == "perf_stat":
            mean_res[k] = _aggregate_perf_stats(v, rounds)

        elif k in ("buy_and_hold", "daily_price_return"):
            mean_res[k] = np.nanmean(v, axis=0)

        else:
            _add_summary_stats(mean_res, k, v)

    # Buy-and-hold baseline perf stats (full test window, no randomisation)
    observation, _ = test_env.reset()
    open_col = f"adjopen_{test_env.symbol}"
    if open_col in test_env.df.columns:
        bnh_returns = (
            test_env.df[open_col]
            .iloc[test_env.frame_bound[0] : test_env.frame_bound[1]]
            .pct_change()
            .fillna(0)
        )
        mean_res["bnh_perf_stat"] = perf_stats_series(
            bnh_returns.values,
            bnh_returns.index,
        )

    return mean_res


# ------------------------------------------------------------------
# Internal aggregation helpers
# ------------------------------------------------------------------


def _aggregate_perf_stats(v: pd.DataFrame, rounds: int) -> pd.DataFrame:
    m = v.mean(axis=1)
    s = v.std(axis=1)
    se = st.sem(v.values, axis=1)
    dof = rounds - 1

    lo95, hi95 = st.t.interval(confidence=0.95, df=dof, loc=m, scale=se)
    lo99, hi99 = st.t.interval(confidence=0.99, df=dof, loc=m, scale=se)

    lo95 = pd.Series(lo95, index=v.index)
    hi95 = pd.Series(hi95, index=v.index)
    lo99 = pd.Series(lo99, index=v.index)
    hi99 = pd.Series(hi99, index=v.index)

    lo_1_5s = m - 1.5 * s
    hi_1_5s = m + 1.5 * s

    best5 = pd.Series(np.quantile(v.values, 0.95, axis=1), index=v.index)
    worst5 = pd.Series(np.quantile(v.values, 0.05, axis=1), index=v.index)
    best20 = pd.Series(np.quantile(v.values, 0.80, axis=1), index=v.index)
    worst20 = pd.Series(np.quantile(v.values, 0.20, axis=1), index=v.index)

    return pd.concat(
        [m, lo95, hi95, lo99, hi99, lo_1_5s, hi_1_5s, best5, worst5, best20, worst20],
        axis=1,
    ).rename(
        columns={
            0: "point_est (mean)",
            1: "%95 conf. lower bound",
            2: "%95 conf. upper bound",
            3: "%99 conf. lower bound",
            4: "%99 conf. upper bound",
            5: "1.5sigma lower",
            6: "1.5sigma upper",
            7: "best_5_pctile",
            8: "worst_5_pctile",
            9: "best_20_pctile",
            10: "worst_20_pctile",
        }
    )


def _add_summary_stats(
    out: Dict[str, Any], key: str, v: np.ndarray
) -> None:
    m = np.nanmean(v, axis=0)
    s = np.nanstd(v, axis=0, ddof=1)
    # Count non-NaN values per timestep for proper SEM / CI
    n = np.sum(~np.isnan(v), axis=0)
    se = s / np.sqrt(n)
    dof = np.maximum(n - 1, 1)

    lo95, hi95 = st.t.interval(confidence=0.95, df=dof, loc=m, scale=se)
    lo99, hi99 = st.t.interval(confidence=0.99, df=dof, loc=m, scale=se)

    out[f"{key}_point"] = m
    out[f"{key}_lower95"] = lo95
    out[f"{key}_upper95"] = hi95
    out[f"{key}_lower99"] = lo99
    out[f"{key}_upper99"] = hi99
    out[f"{key}_min"] = np.nanmin(v, axis=0)
    out[f"{key}_max"] = np.nanmax(v, axis=0)
    out[f"{key}_best5pctile"] = np.nanquantile(v, 0.95, axis=0)
    out[f"{key}_worst5pctile"] = np.nanquantile(v, 0.05, axis=0)
    out[f"{key}_best20pctile"] = np.nanquantile(v, 0.80, axis=0)
    out[f"{key}_worst20pctile"] = np.nanquantile(v, 0.20, axis=0)
    out[f"{key}_lower1_5S"] = m - 1.5 * s
    out[f"{key}_upper1_5S"] = m + 1.5 * s
