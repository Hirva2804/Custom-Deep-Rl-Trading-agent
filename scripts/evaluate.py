#!/usr/bin/env python
"""
Standalone evaluation script.

Usage::

    python scripts/evaluate.py --config configs/ppo_goog.yaml \\
                               --checkpoint checkpoints/ppo/best_model.zip
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO, DQN

from src.utils.config import load_config
from src.utils.helpers import seed_everything
from src.utils.logging import setup_logging
from src.data.source import DataSource
from src.data.preprocess import preprocess_from_config
from src.envs.trading_env import TradingEnv
from src.evaluation.testing import test_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained agent")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a saved model .zip")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--output", type=str, default=None,
                   help="Where to save results .pkl (default: outputs/<agent>_<ticker>_results.pkl)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override or [])
    log = setup_logging()

    seed_everything(cfg["seed"])

    agent_type = cfg["agent"]["type"]
    ticker = cfg["data"]["tickers"][0]
    window_size = cfg["env"]["window_size"]

    # Load data + preprocess
    ds = DataSource(cfg)
    _, _, _, df_test_norm = preprocess_from_config(ds.data.copy(), ticker, cfg)

    reward_cfg = cfg["env"].get("reward", {})
    reward_type = reward_cfg.get("type", "simple_return")
    reward_eta = reward_cfg.get("eta", 0.01)

    test_env = TradingEnv(
        df=df_test_norm,
        window_size=window_size,
        frame_bound=(window_size, len(df_test_norm)),
        init_cash=cfg["env"]["init_cash"],
        symbol=ticker,
        commission_rate=cfg["env"]["commission_rate"],
        reward_type=reward_type,
        reward_eta=reward_eta,
    )

    # Load model
    AgentCls = PPO if agent_type == "PPO" else DQN
    model = AgentCls.load(args.checkpoint)
    log.info("Loaded %s from %s", agent_type, args.checkpoint)

    # Evaluate
    eval_cfg = cfg.get("evaluation", {})
    results = test_model(
        model, test_env,
        rounds=eval_cfg.get("rounds", 500),
        random_start=eval_cfg.get("random_start", True),
    )

    # Save
    output_dir = Path(cfg.get("paths", {}).get("output_dir", "./outputs")) / agent_type.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else output_dir / f"{agent_type}_{ticker}_results.pkl"

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    if "perf_stat" in results and results["perf_stat"] is not None:
        csv_path = out_path.with_suffix(".csv")
        results["perf_stat"].to_csv(csv_path)
        log.info("Perf stats → %s", csv_path)

    log.info("Results → %s", out_path)


if __name__ == "__main__":
    main()
