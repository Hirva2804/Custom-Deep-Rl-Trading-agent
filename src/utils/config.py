"""
Configuration loading, merging, and validation.

Loads the base ``configs/default.yaml``, deep-merges any experiment-specific
YAML on top, applies CLI ``--override`` key=value pairs, and returns a plain
dict ready for consumption by every other module.

API keys are loaded from ``.env`` via *python-dotenv* — never stored in YAML.
"""

from __future__ import annotations

import copy
import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env once at import time so every module can use os.environ
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # …/DeepRL-trade
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into a deep-copy of *base*."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_dates(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert date strings in ``data`` section to ``datetime.datetime``."""
    data = cfg.get("data", {})
    for key in ("start_date", "end_date"):
        val = data.get(key)
        if isinstance(val, str):
            data[key] = datetime.datetime.strptime(val, "%Y-%m-%d")
    return cfg


def _apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Apply dot-separated ``key=value`` overrides from the CLI.

    Examples::

        --override seed=123
        --override agent.type=DQN
        --override agent.ppo.learning_rate=0.001
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key_path, raw_value = item.split("=", 1)
        keys = key_path.split(".")

        # Auto-cast value
        value: Any
        if raw_value.lower() in ("true", "false"):
            value = raw_value.lower() == "true"
        elif raw_value.lower() == "null" or raw_value.lower() == "none":
            value = None
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value  # keep as string

        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    return cfg


def _validate(cfg: Dict[str, Any]) -> None:
    """Minimal validation of required fields."""
    assert "seed" in cfg, "Config must contain 'seed'"
    assert "data" in cfg, "Config must contain 'data' section"
    assert "env" in cfg, "Config must contain 'env' section"
    assert "agent" in cfg, "Config must contain 'agent' section"
    agent_type = cfg["agent"]["type"]
    assert agent_type in ("PPO", "DQN"), f"Unsupported agent type: {agent_type}"
    data_src = cfg["data"].get("source", "yfinance")
    if data_src == "tiingo" and not os.environ.get("TIINGO_API_KEY"):
        raise EnvironmentError(
            "data.source is 'tiingo' but TIINGO_API_KEY is not set in .env"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load and return the merged experiment configuration.

    Parameters
    ----------
    config_path : str | None
        Path to an experiment YAML that overrides ``configs/default.yaml``.
        If *None*, only the defaults are used.
    overrides : list[str] | None
        List of ``key=value`` strings for CLI overrides.

    Returns
    -------
    dict
        Fully merged, date-parsed, validated configuration dictionary.
    """
    # 1. Load base defaults
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Merge experiment-specific overrides
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            exp_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, exp_cfg)

    # 3. Apply CLI overrides
    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    # 4. Parse dates
    cfg = _parse_dates(cfg)

    # 5. Validate
    _validate(cfg)

    return cfg
