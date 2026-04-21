"""
Visualisation functions for experiment results.

Every function accepts an optional ``save_path``; when provided the figure
is saved to disk (useful for automated experiment pipelines).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ------------------------------------------------------------------
# Style
# ------------------------------------------------------------------
def _apply_style() -> None:
    plt.style.use("seaborn-v0_8")


# ------------------------------------------------------------------
# 1. Test-result cumulative-return plot (with confidence bands)
# ------------------------------------------------------------------

def plot_test_result(
    result_dict: Dict[str, Any],
    model_name: str,
    symbol: str,
    split: str = "Test",
    *,
    plot_95_conf: bool = True,
    plot_99_conf: bool = False,
    plot_1_5s: bool = True,
    plot_minmax: bool = False,
    plot_bestworst20: bool = False,
    plot_bestworst5: bool = False,
    save_path: Optional[str] = None,
) -> None:
    """Plot cumulative return with optional confidence bands."""
    _apply_style()
    dates = pd.to_datetime(result_dict["date"])

    plt.figure(figsize=(16, 10))
    plt.title(f"{model_name} {split} Performance — {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.plot(dates, result_dict["buy_and_hold"], label="Buy and Hold")
    plt.plot(
        dates, result_dict["total_profit_percentage_point"],
        label=model_name, color="r",
    )

    if plot_95_conf:
        plt.fill_between(
            dates,
            result_dict["total_profit_percentage_upper95"],
            result_dict["total_profit_percentage_lower95"],
            label="95% Conf. Int.", alpha=0.8,
        )
    if plot_99_conf:
        plt.fill_between(
            dates,
            result_dict["total_profit_percentage_upper99"],
            result_dict["total_profit_percentage_lower99"],
            label="99% Conf. Int.", alpha=0.6,
        )
    if plot_1_5s:
        plt.fill_between(
            dates,
            result_dict["total_profit_percentage_upper1_5S"],
            result_dict["total_profit_percentage_lower1_5S"],
            label="1.5 Sigma", alpha=0.65,
        )
    if plot_minmax:
        plt.fill_between(
            dates,
            result_dict["total_profit_percentage_max"],
            result_dict["total_profit_percentage_min"],
            label="Min/Max", alpha=0.3,
        )
    if plot_bestworst20:
        plt.plot(dates, result_dict["total_profit_best20pctile"],
                 label="Best 20th pctile", alpha=0.5)
        plt.plot(dates, result_dict["total_profit_worst20pctile"],
                 label="Worst 20th pctile", alpha=0.5)
    if plot_bestworst5:
        plt.plot(dates, result_dict["total_profit_best5pctile"],
                 label="Best 5th pctile", alpha=0.5)
        plt.plot(dates, result_dict["total_profit_worst5pctile"],
                 label="Worst 5th pctile", alpha=0.5)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    _maybe_save(save_path)


# ------------------------------------------------------------------
# 2. Price chart with buy/sell markers
# ------------------------------------------------------------------

def plot_price_chart(
    env,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Render the full episode from *env* (after running an episode)."""
    _apply_style()
    plt.figure(figsize=(16, 8))
    env.render_all()
    if title:
        plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Open Price")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    _maybe_save(save_path)


# ------------------------------------------------------------------
# 3. Position history
# ------------------------------------------------------------------

def plot_position_history(
    env,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> None:
    _apply_style()
    dates = pd.to_datetime(env.history["date"])
    positions = env.get_history_slice("position_history")

    plt.figure(figsize=(20, 8))
    plt.step(dates, positions, label="Position Value", linewidth=0.8)
    plt.title(f"{model_name} Position History")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    _maybe_save(save_path)


# ------------------------------------------------------------------
# 4. Action history
# ------------------------------------------------------------------

def plot_action_history(
    env,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> None:
    _apply_style()
    dates = pd.to_datetime(env.history["date"])
    actions = env.get_history_slice("actions_history")

    plt.figure(figsize=(20, 8))
    plt.step(dates, actions, label="Actions", linewidth=0.9)
    plt.title(f"{model_name} Actions History")
    plt.xlabel("Date")
    plt.ylabel("Action")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    _maybe_save(save_path)


# ------------------------------------------------------------------
# 5. Distribution plots (KDE / histogram / boxplot)
# ------------------------------------------------------------------

def plot_daily_returns_kde(
    daily_rets: pd.DataFrame,
    title: str = "Daily Returns Distribution",
    save_path: Optional[str] = None,
) -> None:
    _apply_style()
    plt.figure(figsize=(16, 10))
    ax = sns.kdeplot(data=daily_rets, fill=True, common_norm=False,
                     alpha=0.2, linewidth=1.3)
    sns.move_legend(ax, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.xlabel("Daily Returns")
    plt.ylabel("Density")
    plt.tight_layout()
    _maybe_save(save_path)


def plot_daily_returns_hist(
    daily_rets: pd.DataFrame,
    title: str = "Daily Returns Histogram",
    save_path: Optional[str] = None,
) -> None:
    _apply_style()
    plt.figure(figsize=(16, 10))
    ax = sns.histplot(data=daily_rets, fill=True, alpha=0.5, linewidth=1.3)
    sns.move_legend(ax, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    _maybe_save(save_path)


def plot_daily_returns_boxplot(
    daily_rets: pd.DataFrame,
    title: str = "Daily Returns Boxplot",
    save_path: Optional[str] = None,
) -> None:
    _apply_style()
    plt.figure(figsize=(16, 6))
    sns.boxplot(data=daily_rets)
    plt.title(title)
    plt.xlabel("Agent")
    plt.ylabel("Return")
    plt.tight_layout()
    _maybe_save(save_path)


# ------------------------------------------------------------------
# 6. Position/action value distributions
# ------------------------------------------------------------------

def plot_value_distribution(
    values: np.ndarray,
    dates,
    label: str = "Values",
    title: str = "Distribution",
    save_path: Optional[str] = None,
) -> None:
    _apply_style()
    df = pd.DataFrame({label: values}, index=pd.to_datetime(dates))
    plt.figure(figsize=(16, 10))
    ax = sns.histplot(data=df, fill=True, bins=50, alpha=0.7,
                      linewidth=1.5, kde=True)
    sns.move_legend(ax, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.xlabel(label)
    plt.tight_layout()
    _maybe_save(save_path)


# ------------------------------------------------------------------
# 7. Feature correlation heatmap
# ------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Feature Correlation Matrix",
    drop_columns: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    _apply_style()
    if drop_columns:
        df = df.drop(columns=drop_columns, errors="ignore")
    corr = df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=corr.values, cmap="RdYlGn", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    _maybe_save(save_path)


# ------------------------------------------------------------------
# 8. Multi-strategy comparison
# ------------------------------------------------------------------

def plot_comparison(
    data: pd.DataFrame,
    title: str = "Strategies Performance",
    ylabel: str = "Cumulative Return",
    save_path: Optional[str] = None,
) -> None:
    """Line plot comparing multiple strategies over time (index = dates)."""
    _apply_style()
    plt.figure(figsize=(16, 8))
    ax = sns.lineplot(data=data)
    sns.move_legend(ax, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.tight_layout()
    _maybe_save(save_path)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _maybe_save(path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
