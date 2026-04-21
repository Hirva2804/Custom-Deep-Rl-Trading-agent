"""
DQN agent factory.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from stable_baselines3 import DQN

log = logging.getLogger("deeprl_trade.agents")


def build_dqn_agent(
    env,
    config: Dict[str, Any],
    tensorboard_log: Optional[str] = None,
) -> DQN:
    """
    Instantiate a DQN agent from the experiment config.

    Parameters
    ----------
    env : gymnasium.Env
        The training environment.
    config : dict
        Full experiment configuration.
    tensorboard_log : str | None
        Path for TensorBoard / wandb logging.

    Returns
    -------
    DQN
    """
    dqn_cfg = config["agent"]["dqn"]

    agent = DQN(
        "MlpPolicy",
        env,
        learning_rate=dqn_cfg.get("learning_rate", 1e-4),
        buffer_size=dqn_cfg.get("buffer_size", 1_000_000),
        learning_starts=dqn_cfg.get("learning_starts", 50_000),
        batch_size=dqn_cfg.get("batch_size", 512),
        tau=dqn_cfg.get("tau", 0.99),
        gamma=dqn_cfg.get("gamma", 0.99),
        train_freq=(dqn_cfg.get("train_freq", 1000), "step"),
        gradient_steps=1,
        target_update_interval=dqn_cfg.get("target_update_interval", 5000),
        exploration_fraction=dqn_cfg.get("exploration_fraction", 0.2),
        exploration_initial_eps=dqn_cfg.get("exploration_initial_eps", 1.0),
        exploration_final_eps=dqn_cfg.get("exploration_final_eps", 0.02),
        verbose=1,
        seed=config.get("seed", 42),
        tensorboard_log=tensorboard_log,
    )

    log.info("Built DQN agent  (lr=%.1e, buffer=%d, batch=%d)",
             dqn_cfg.get("learning_rate", 1e-4),
             dqn_cfg.get("buffer_size", 1_000_000),
             dqn_cfg.get("batch_size", 512))
    return agent
