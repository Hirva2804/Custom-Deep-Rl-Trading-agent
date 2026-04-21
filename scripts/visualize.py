#!/usr/bin/env python
"""
Generate all visualisations from saved evaluation results.

Usage::

    python scripts/visualize.py --results outputs/ppo/PPO_GOOG_results.pkl \\
                                --output figures/
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.visualization.plots import (
    plot_test_result,
    plot_daily_returns_kde,
    plot_daily_returns_hist,
    plot_daily_returns_boxplot,
    plot_comparison,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate plots from saved results")
    p.add_argument("--results", nargs="+", required=True,
                   help="One or more .pkl result files")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Label for each result file (default: derived from filename)")
    p.add_argument("--output", type=str, default="./figures",
                   help="Directory for saved figures")
    p.add_argument("--symbol", type=str, default="GOOG")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    labels = args.labels or []
    for i, rpath in enumerate(args.results):
        with open(rpath, "rb") as f:
            res = pickle.load(f)
        label = labels[i] if i < len(labels) else Path(rpath).stem.split("_")[0]
        results.append((label, res))

    # 1. Per-agent cumulative return plots
    for label, res in results:
        plot_test_result(
            res, model_name=label, symbol=args.symbol,
            save_path=str(out / f"{label}_cumulative_return.png"),
        )

    # 2. Daily returns comparison (if ≥1 result)
    daily_rets = pd.DataFrame()
    for label, res in results:
        key = "daily_return_point"
        if key in res:
            daily_rets[label] = res[key]
    if "buy_and_hold" in results[0][1]:
        # Use daily_price_return as Buy&Hold proxy
        key = "daily_price_return"
        if key in results[0][1]:
            daily_rets["Buy&Hold"] = results[0][1][key]

    if not daily_rets.empty:
        plot_daily_returns_kde(
            daily_rets,
            title=f"Daily Returns Distribution ({args.symbol})",
            save_path=str(out / "daily_returns_kde.png"),
        )
        plot_daily_returns_hist(
            daily_rets,
            title=f"Daily Returns Histogram ({args.symbol})",
            save_path=str(out / "daily_returns_hist.png"),
        )
        plot_daily_returns_boxplot(
            daily_rets,
            title=f"Daily Returns Boxplot ({args.symbol})",
            save_path=str(out / "daily_returns_boxplot.png"),
        )

    # 3. Cumulative return comparison line plot
    comparison = pd.DataFrame()
    for label, res in results:
        key = "total_profit_percentage_point"
        if key in res and "date" in res:
            comparison[label] = res[key]
    if "buy_and_hold" in results[0][1] and "date" in results[0][1]:
        comparison["Buy&Hold"] = results[0][1]["buy_and_hold"]
        comparison.index = pd.to_datetime(results[0][1]["date"])

    if not comparison.empty:
        plot_comparison(
            comparison,
            title=f"Strategies Average Performance ({args.symbol})",
            save_path=str(out / "strategies_comparison.png"),
        )

    print(f"Figures saved to {out.resolve()}")


if __name__ == "__main__":
    main()
