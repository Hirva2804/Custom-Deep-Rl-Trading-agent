#!/usr/bin/env python
"""
Training entry point.

Usage::

    python scripts/train.py --config configs/ppo_goog.yaml
    python scripts/train.py --config configs/dqn_goog.yaml --override seed=123
    python scripts/train.py  # uses configs/default.yaml
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3.common.callbacks import EvalCallback

from src.utils.config import load_config
from src.utils.helpers import seed_everything
from src.utils.logging import setup_logging
from src.data.source import DataSource
from src.data.preprocess import preprocess_from_config
from src.envs.trading_env import TradingEnv
from src.agents.ppo import build_ppo_agent
from src.agents.dqn import build_dqn_agent
from src.evaluation.testing import test_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DeepRL trading agent")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to experiment YAML (merged on top of configs/default.yaml)",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="key=value overrides, e.g. --override seed=123 agent.type=DQN",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override or [])
    log = setup_logging()

    seed = cfg["seed"]
    seed_everything(seed)

    agent_type = cfg["agent"]["type"]
    tickers = cfg["data"]["tickers"]
    window_size = cfg["env"]["window_size"]

    # ----- Paths -------------------------------------------------------
    paths = cfg.get("paths", {})
    checkpoint_dir = Path(paths.get("checkpoint_dir", "./checkpoints")) / agent_type.lower()
    log_dir = Path(paths.get("log_dir", "./logs")) / agent_type.lower()
    output_dir = Path(paths.get("output_dir", "./outputs")) / agent_type.lower()
    for d in (checkpoint_dir, log_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ----- wandb -------------------------------------------------------
    tracking = cfg.get("tracking", {})
    wandb_run = None
    wandb_callback = None

    if tracking.get("use_wandb", False):
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback

            wandb_run = wandb.init(
                project=tracking.get("project", "deeprl-trade"),
                entity=tracking.get("entity"),
                config=cfg,
                sync_tensorboard=True,
                save_code=True,
            )
            wandb_callback = WandbCallback(
                model_save_path=str(checkpoint_dir / "wandb"),
                verbose=1,
            )
            log.info("wandb run initialised: %s", wandb_run.url)
        except ImportError:
            log.warning("wandb not installed — skipping experiment tracking")

    # ----- Data --------------------------------------------------------
    ticker = tickers[0]  # primary ticker
    log.info("Loading data for %s …", ticker)

    ds = DataSource(cfg)
    _df_train, _df_test, df_train_norm, df_test_norm = preprocess_from_config(
        ds.data.copy(), ticker, cfg,
    )

    # ----- Environments ------------------------------------------------
    reward_cfg = cfg["env"].get("reward", {})
    reward_type = reward_cfg.get("type", "simple_return")
    reward_eta = reward_cfg.get("eta", 0.01)

    env = TradingEnv(
        df=df_train_norm,
        window_size=window_size,
        frame_bound=(window_size, len(df_train_norm)),
        init_cash=cfg["env"]["init_cash"],
        symbol=ticker,
        commission_rate=cfg["env"]["commission_rate"],
        reward_type=reward_type,
        reward_eta=reward_eta,
    )
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

    # ----- Build agent -------------------------------------------------
    tb_log = str(log_dir)
    if agent_type == "PPO":
        agent = build_ppo_agent(env, cfg, tensorboard_log=tb_log)
    elif agent_type == "DQN":
        agent = build_dqn_agent(env, cfg, tensorboard_log=tb_log)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # ----- Training ----------------------------------------------------
    total_timesteps = cfg["agent"]["total_timesteps"]
    eval_freq = cfg["agent"]["eval_freq"]

    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=str(checkpoint_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )

    callbacks = [eval_callback]
    if wandb_callback is not None:
        callbacks.append(wandb_callback)

    log.info("Training %s for %d timesteps …", agent_type, total_timesteps)
    agent.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Save final model
    final_path = str(checkpoint_dir / f"{agent_type}_{ticker}_final")
    agent.save(final_path)
    log.info("Final model saved to %s", final_path)

    # ----- Evaluation --------------------------------------------------
    best_path = str(checkpoint_dir / "best_model.zip")
    if Path(best_path).exists():
        if agent_type == "PPO":
            from stable_baselines3 import PPO as AgentCls
        else:
            from stable_baselines3 import DQN as AgentCls
        best_agent = AgentCls.load(best_path)
        log.info("Loaded best model from %s", best_path)
    else:
        best_agent = agent
        log.warning("best_model.zip not found; evaluating final model instead")

    eval_cfg = cfg.get("evaluation", {})
    rounds = eval_cfg.get("rounds", 500)
    random_start = eval_cfg.get("random_start", True)

    log.info("Evaluating on test set (%d rounds) …", rounds)
    results = test_model(best_agent, test_env, rounds=rounds, random_start=random_start)

    # Save results
    results_path = output_dir / f"{agent_type}_{ticker}_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    log.info("Results saved to %s", results_path)

    if "perf_stat" in results and results["perf_stat"] is not None:
        csv_path = output_dir / f"{agent_type}_{ticker}_perf_stats.csv"
        results["perf_stat"].to_csv(csv_path)
        log.info("Performance stats saved to %s", csv_path)

    # Log to wandb
    if wandb_run is not None:
        import wandb

        wandb.save(str(results_path))
        if "perf_stat" in results:
            wandb.log({"perf_stats": wandb.Table(dataframe=results["perf_stat"])})
        wandb_run.finish()

    log.info("Done.")


if __name__ == "__main__":
    main()
