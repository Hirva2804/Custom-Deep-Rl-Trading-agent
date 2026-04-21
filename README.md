# DeepRL-trade

Algorithmic Trading Using Deep Reinforcement Learning (PPO & DQN)

---

## Introduction

In quantitative finance, stock trading is essentially a dynamic decision problem — deciding where, at what price, and how much to trade in a stochastic, dynamic, and complex market. Deep reinforcement learning (DRL) enables modelling and solving these sequential decision problems with a human-like approach.

This project trains two DRL agents — **Proximal Policy Optimization (PPO)** and **Deep Q-Learning (DQN)** — to autonomously make trading decisions on **GOOG** stock and compares their performance against a Buy & Hold benchmark using risk-adjusted metrics.

---

## Project Structure

```
DeepRL-trade/
├── configs/
│   ├── default.yaml          # Base config (all defaults)
│   ├── ppo_goog.yaml         # PPO experiment overrides
│   └── dqn_goog.yaml         # DQN experiment overrides
├── src/
│   ├── data/
│   │   ├── source.py         # DataSource (yfinance + Tiingo backends)
│   │   └── preprocess.py     # Train/test split, rolling z-score normalisation
│   ├── envs/
│   │   └── trading_env.py    # Gymnasium-based trading environment
│   ├── agents/
│   │   ├── networks.py       # Custom actor-critic network for PPO
│   │   ├── ppo.py            # PPO agent factory
│   │   └── dqn.py            # DQN agent factory
│   ├── evaluation/
│   │   ├── testing.py        # Multi-round evaluation with statistical analysis
│   │   └── metrics.py        # pyfolio perf_stats wrapper, CI helpers
│   ├── visualization/
│   │   └── plots.py          # All plotting functions
│   └── utils/
│       ├── config.py         # YAML config loading, merging, validation
│       ├── helpers.py         # Seed, rounding, index helpers
│       └── logging.py        # Centralised logging
├── scripts/
│   ├── train.py              # Training entry point
│   ├── evaluate.py           # Standalone evaluation
│   └── visualize.py          # Generate figures from saved results
├── notebooks/
│   └── DeepRL_trader.ipynb   # Original notebook (reference)
├── configs/
├── requirements.txt
├── .env.example              # API key template
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### 2. Configure API keys (optional)

```bash
cp .env.example .env
# Edit .env and add your TIINGO_API_KEY (only needed if using Tiingo)
```

### 3. Train

```bash
# Train PPO agent on GOOG (default config)
python scripts/train.py --config configs/ppo_goog.yaml

# Train DQN agent
python scripts/train.py --config configs/dqn_goog.yaml

# Override any parameter from CLI
python scripts/train.py --config configs/ppo_goog.yaml --override seed=123 agent.total_timesteps=100000
```

### 4. Evaluate

```bash
python scripts/evaluate.py --config configs/ppo_goog.yaml \
                           --checkpoint checkpoints/ppo/best_model.zip
```

### 5. Visualise

```bash
python scripts/visualize.py --results outputs/ppo/PPO_GOOG_results.pkl \
                                      outputs/dqn/DQN_GOOG_results.pkl \
                            --output figures/
```

---

## Configuration

All settings are in YAML files under `configs/`. The system works in layers:

1. **`configs/default.yaml`** — all defaults
2. **Experiment YAML** (e.g. `ppo_goog.yaml`) — overrides specific keys
3. **CLI `--override`** — overrides anything at runtime

Key sections: `data`, `env`, `agent` (with `ppo`/`dqn` sub-sections), `evaluation`, `tracking`, `paths`.

### Experiment Tracking

Set `tracking.use_wandb: true` in your config and ensure `WANDB_API_KEY` is in `.env` (or run `wandb login`). Metrics, models, and configs are auto-logged to your Weights & Biases project.

---

## How It Works

| Component | Description |
|-----------|-------------|
| **Environment** | Gymnasium-compatible discrete-action env. Actions: Long (+1), Cash (0), Short (-1). Reward = % change in total assets. |
| **PPO Agent** | Custom actor-critic with LayerNorm + Dropout (configurable architecture). |
| **DQN Agent** | Standard MLP policy from stable-baselines3. |
| **Evaluation** | 500 random-start episodes → point estimates, 95%/99% CIs, percentile bands, pyfolio stats. |
| **Data** | OHLCV from yfinance (default) or Tiingo. Technical indicators via pandas-ta. Rolling z-score normalisation. |

---

## References

- Human-level control through deep reinforcement learning (DQN): [paper](https://www.nature.com/articles/nature14236)
- Proximal Policy Optimization (PPO): [paper](https://arxiv.org/abs/1707.06347), [blog](https://openai.com/blog/openai-baselines-ppo/), [spinning-up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
