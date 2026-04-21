"""
PPO agent factory.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from stable_baselines3 import PPO

from src.agents.networks import make_policy_class

log = logging.getLogger("deeprl_trade.agents")


def build_ppo_agent(
    env,
    config: Dict[str, Any],
    tensorboard_log: Optional[str] = None,
) -> PPO:
    """
    Instantiate a PPO agent from the experiment config.

    Parameters
    ----------
    env : gymnasium.Env
        The training environment (or ``DummyVecEnv`` wrapper).
    config : dict
        Full experiment configuration.
    tensorboard_log : str | None
        Path for TensorBoard / wandb logging.

    Returns
    -------
    PPO
    """
    ppo_cfg = config["agent"]["ppo"]
    net_cfg = ppo_cfg.get("network", {})

    policy_cls = make_policy_class(
        hidden_dim=net_cfg.get("hidden_dim", 256),
        output_dim=net_cfg.get("output_dim", 64),
        dropout=net_cfg.get("dropout", 0.07),
    )

    agent = PPO(
        policy_cls,
        env,
        learning_rate=ppo_cfg.get("learning_rate", 1e-4),
        n_steps=ppo_cfg.get("n_steps", 1024),
        batch_size=ppo_cfg.get("batch_size", 128),
        n_epochs=ppo_cfg.get("n_epochs", 15),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        clip_range_vf=None,
        normalize_advantage=True,
        verbose=1,
        seed=config.get("seed", 42),
        tensorboard_log=tensorboard_log,
    )

    log.info("Built PPO agent  (lr=%.1e, steps=%d, epochs=%d)",
             ppo_cfg.get("learning_rate", 1e-4),
             ppo_cfg.get("n_steps", 1024),
             ppo_cfg.get("n_epochs", 15))
    return agent
