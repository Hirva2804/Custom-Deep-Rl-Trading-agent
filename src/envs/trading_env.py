"""
Custom trading environment with the **gymnasium** interface.

Migrated from the legacy ``gym.Env`` in the original notebook. Key changes:

* ``step()`` returns the gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``
* ``reset()`` accepts ``seed`` and ``options`` per gymnasium API
* ``init_cash`` and ``commission_rate`` are passed explicitly (Bug #4, #5)
* Seed propagation for reproducibility (Bug #15)
* ``render_all()`` alignment fix (Bug #10)
* No hardcoded ``[11:]`` slicing — uses ``window_size + 1`` (Bug #11)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.utils.helpers import rnd

log = logging.getLogger("deeprl_trade.envs")


class TradingEnv(gym.Env):
    """
    A discrete-action trading environment.

    Actions
    -------
    0 → short (-1),  1 → cash (0),  2 → long (+1)

    Observation
    -----------
    Flattened window of signal features concatenated with two portfolio-state
    scalars (cash ratio, position ratio).
    """

    metadata = {"render_modes": ["human"]}

    # Supported reward function types
    REWARD_TYPES = ("simple_return", "dsr")

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        frame_bound: Tuple[int, int],
        init_cash: float = 1_000_000,
        symbol: str = "",
        commission_rate: float = 0.001,
        reward_type: str = "simple_return",
        reward_eta: float = 0.01,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert df.ndim == 2

        if reward_type not in self.REWARD_TYPES:
            raise ValueError(
                f"Unknown reward_type={reward_type!r}. "
                f"Choose from {self.REWARD_TYPES}"
            )

        self.df = df
        self.symbol = symbol
        self.commission_rate = commission_rate
        self.frame_bound = frame_bound
        self.window_size = window_size
        self.render_mode = render_mode
        self.init_cash = init_cash
        self.reward_type = reward_type
        self.reward_eta = reward_eta

        self.open_prices, self.close_prices, self.signal_features, self.dates = self._process_data()

        # observation = window of signal features + 2 portfolio scalars
        self.shape = (window_size * self.signal_features.shape[1] + 2,)

        self.action_space = spaces.Discrete(n=3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64
        )

        # Episode state (initialised in reset)
        self._start_tick: int = self.window_size
        self._end_tick: int = len(self.open_prices) - 1
        self.starting_price: float = 0.0
        self._done: bool = True
        self._current_tick: int = 0
        self._last_trade_tick: int = 0
        self._cash: float = init_cash
        self._position_value: float = 0.0
        self._total_assets: float = init_cash
        self._init_total_assets: float = init_cash
        self._last_trade: float = 0.0
        self._position_history: list = []
        self._trade_history: list = []
        self._actions_history: list = []
        self._total_reward: float = 0.0
        self._total_profit: float = 0.0
        self._total_pct_profit: float = 0.0
        self._first_rendering: bool = True
        self.curr_open_price: float = 0.0
        self.history: Dict[str, list] = {}

        # DSR running statistics (initialised properly in reset)
        self._dsr_A: float = 0.0  # EMA of returns
        self._dsr_B: float = 0.0  # EMA of squared returns

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return ``(observation, info)``."""
        super().reset(seed=seed)

        rand_trail = (options or {}).get("rand_trail", False)

        self._done = False
        if rand_trail:
            self._start_tick = int(self.np_random.integers(
                self.frame_bound[0], self.frame_bound[1] - 20
            ))
        else:
            self._start_tick = self.frame_bound[0]
        self._end_tick = self.frame_bound[1] - 1
        self._current_tick = self._start_tick
        self.starting_price = self.open_prices[self._start_tick]
        self.curr_open_price = self.open_prices[self._start_tick]
        self._last_trade_tick = self._current_tick - 1

        self._cash = self.init_cash
        self._position_value = 0.0
        self._total_assets = self._cash + self._position_value
        self._init_total_assets = self._total_assets
        self._last_trade = 0.0

        self._position_history = [0] * self.window_size + [self._position_value]
        self._trade_history = [0] * self.window_size + [self._last_trade]
        self._actions_history = [0] * (1 + self.window_size)

        self._total_reward = 0.0
        self._total_profit = 0.0
        self._total_pct_profit = 0.0
        self._first_rendering = True
        self.history = {}

        # Reset DSR running statistics
        self._dsr_A = 0.0
        self._dsr_B = 0.0

        obs = self._get_observation()
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading step.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        self._current_tick += 1
        self.curr_open_price = self.open_prices[self._current_tick]

        # Map discrete {0, 1, 2} → {-1, 0, +1}
        mapped_action = action - 1

        terminated = False
        truncated = False

        if self._current_tick >= self._end_tick:
            truncated = True

        last_day_total_assets = self._total_assets
        step_reward = self._calculate_reward(mapped_action)

        # Terminate if portfolio is essentially wiped out
        if self._total_assets <= 0:
            terminated = True
            step_reward = -1.0

        self._actions_history.append(mapped_action)
        self._total_reward += step_reward
        self._total_profit = self._total_assets - self.init_cash
        self._total_pct_profit = self._total_profit / self.init_cash

        observation = self._get_observation()
        info = dict(
            date=self.dates[self._current_tick],
            total_reward=rnd(self._total_reward),
            total_profit=rnd(self._total_profit),
            total_profit_percentage=rnd(self._total_pct_profit),
            buy_and_hold=rnd((self.curr_open_price / self.starting_price) - 1),
            daily_price_return=rnd(
                (self.curr_open_price / self.open_prices[self._current_tick - 1]) - 1
            ),
            daily_return=rnd(
                (self._total_assets / last_day_total_assets) - 1
                if last_day_total_assets > 0
                else 0.0
            ),
            total_assets=rnd(self._total_assets),
        )
        self._update_history(info)

        return observation, step_reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        sig = self.signal_features[
            (self._current_tick - self.window_size) : self._current_tick
        ]
        cash_ratio = self._cash / self._total_assets if self._total_assets > 0 else 1.0
        position_ratio = (
            self._position_value / self._total_assets if self._total_assets > 0 else 0.0
        )
        portfolio_state = np.array([cash_ratio, position_ratio])
        return np.concatenate([sig.reshape(-1), portfolio_state])

    # ------------------------------------------------------------------
    # Reward / trading logic
    # ------------------------------------------------------------------

    def _calculate_reward(self, action: int) -> float:
        """Execute the trade and return the reward for this step."""
        current_price = self.open_prices[self._current_tick]
        last_day_price = self.open_prices[self._current_tick - 1]
        pct_change = current_price / last_day_price

        before_total_assets = self._total_assets

        # Update position value with price change
        self._position_value *= pct_change
        self._total_assets = self._cash + self._position_value

        total_assets = self._total_assets

        # Determine the current directional sign of the position:
        #   +1 if long, -1 if short, 0 if flat
        current_sign = int(np.sign(self._position_value))

        # Only rebalance when the target allocation changes direction
        if action == current_sign:
            # Agent wants to stay in same allocation — no trade needed
            num_assets_to_trade = 0
            actual_position_value_diff = 0.0
            actual_new_position_value = self._position_value
            transaction_cost = 0.0
        else:
            # Budget for transaction costs so cash never goes negative
            if action != 0:
                allocatable = total_assets / (1 + self.commission_rate)
            else:
                allocatable = 0.0
            new_position_value = action * allocatable

            position_value_diff = new_position_value - self._position_value
            # Use int() (truncation toward zero)
            num_assets_to_trade = int(position_value_diff / current_price)
            actual_position_value_diff = num_assets_to_trade * current_price
            actual_new_position_value = actual_position_value_diff + self._position_value

            # Transaction costs
            transaction_cost = abs(actual_position_value_diff) * self.commission_rate

            self._position_value = actual_new_position_value
            self._cash = total_assets - actual_new_position_value - transaction_cost
            self._total_assets = self._cash + self._position_value

        self._last_trade = actual_position_value_diff
        self._trade_history.append(num_assets_to_trade)
        self._position_history.append(actual_new_position_value)
        self._last_trade_tick = self._current_tick

        # single-step portfolio return (used by all reward types)
        R_t = (
            (self._total_assets - before_total_assets) / before_total_assets
            if before_total_assets > 0
            else 0.0
        )

        # Dispatch to the selected reward function
        if self.reward_type == "dsr":
            return self._reward_dsr(R_t)
        # default: simple_return
        return R_t

    # ------------------------------------------------------------------
    # Reward functions
    # ------------------------------------------------------------------

    def _reward_dsr(self, R_t: float) -> float:
        """
        Differential Sharpe Ratio (Moody & Saffell, 2001).

        Provides an online, incremental approximation of the change in Sharpe
        ratio when a new return *R_t* is observed.  Uses exponential moving
        averages of returns (*A*) and squared returns (*B*) with adaptation
        rate ``self.reward_eta``.

        .. math::

            \\Delta A_t = R_t - A_{t-1}
            \\Delta B_t = R_t^2 - B_{t-1}
            D_t = \\frac{B_{t-1} \\Delta A_t
                        - \\tfrac{1}{2} A_{t-1} \\Delta B_t}
                       {(B_{t-1} - A_{t-1}^2)^{3/2}}

        On the first step (when *B − A²* is near zero) the method falls back
        to the raw return to avoid division by zero.
        """
        eta = self.reward_eta

        delta_A = R_t - self._dsr_A
        delta_B = R_t ** 2 - self._dsr_B

        variance_term = self._dsr_B - self._dsr_A ** 2

        if variance_term > 1e-12:
            dsr = (
                self._dsr_B * delta_A - 0.5 * self._dsr_A * delta_B
            ) / (variance_term ** 1.5)
        else:
            # Not enough history to estimate variance — fall back to raw return
            dsr = R_t

        # Update running statistics
        self._dsr_A += eta * delta_A
        self._dsr_B += eta * delta_B

        return dsr

    # ------------------------------------------------------------------
    # Data processing
    # ------------------------------------------------------------------

    def _process_data(self):
        close_col = f"adjclose_{self.symbol}"
        open_col = f"adjopen_{self.symbol}"

        close_prices = self.df[close_col].to_numpy()
        open_prices = self.df[open_col].to_numpy()

        # Validate frame bounds
        _ = open_prices[self.frame_bound[0] - self.window_size]

        sl = slice(self.frame_bound[0] - self.window_size, self.frame_bound[1])
        close_prices = close_prices[sl]
        open_prices = open_prices[sl]

        # Build feature matrix — prefer normalised columns
        base_features = ["adjvolume", "ret_5", "ret_10", "ret_21", "rsi", "macd", "atr"]
        features: list[str] = []
        has_raw_fallback = False
        for feat in base_features:
            norm_col = f"{feat}_{self.symbol}_norm"
            raw_col = f"{feat}_{self.symbol}"
            if norm_col in self.df.columns:
                features.append(norm_col)
            elif raw_col in self.df.columns:
                features.append(raw_col)
                has_raw_fallback = True

        if has_raw_fallback:
            warnings.warn(
                "Some features are using raw (unnormalised) columns. "
                "This may cause feature-scale mismatch. "
                "Consider passing normalised data.",
                stacklevel=2,
            )

        signal_features = self.df[features].values[sl]
        dates = self.df.index[sl]
        return open_prices, close_prices, signal_features, dates

    # ------------------------------------------------------------------
    # History tracking
    # ------------------------------------------------------------------

    def _update_history(self, info: Dict[str, Any]) -> None:
        if not self.history:
            self.history = {key: [] for key in info}
        for key, value in info.items():
            self.history[key].append(value)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> None:  # pragma: no cover
        def _plot_position(position, tick):
            color = None
            if position > 0:
                color = "red"
            elif position < 0:
                color = "green"
            if color:
                plt.scatter(tick, self.open_prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.open_prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position_value, self._current_tick)
        plt.suptitle(
            f"Total Reward: {self._total_reward:.6f} ~ "
            f"Total Profit: {self._total_profit:.6f}"
        )
        plt.pause(0.01)

    def render_all(self) -> None:  # pragma: no cover
        """Plot the full episode with buy/sell markers."""
        window_ticks = np.arange(self._start_tick, self._end_tick)

        plt.plot(
            pd.to_datetime(self.dates),
            self.open_prices,
            label=f"Open Price - {self.symbol}",
        )

        short_ticks, short_dates = [], []
        long_ticks, long_dates = [], []

        # Bug #10 fix: align trade_history with window_ticks properly
        offset = self.window_size + 1  # leading zeros in _trade_history
        for i, tick in enumerate(window_ticks):
            hist_idx = offset + i
            if hist_idx >= len(self._trade_history):
                break
            trade_val = self._trade_history[hist_idx]
            if trade_val < 0:
                short_ticks.append(tick)
                short_dates.append(self.dates[tick])
            elif trade_val > 0:
                long_ticks.append(tick)
                long_dates.append(self.dates[tick])

        plt.plot(
            pd.to_datetime(short_dates),
            self.open_prices[short_ticks],
            "rv",
            markersize=6,
            label="Sell",
        )
        plt.plot(
            pd.to_datetime(long_dates),
            self.open_prices[long_ticks],
            "g^",
            markersize=6,
            label="Buy",
        )
        plt.suptitle(
            f"Total Reward: {self._total_reward:.6f} ~ "
            f"Total Profit: {self._total_profit:.6f}"
        )

    def close(self) -> None:  # pragma: no cover
        plt.close()

    def save_rendering(self, filepath: str) -> None:  # pragma: no cover
        plt.savefig(filepath)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    def get_history_slice(self, key: str) -> list:
        """Return history list aligned *after* the window padding."""
        offset = self.window_size + 1
        attr = getattr(self, f"_{key}", None)
        if attr is not None:
            return attr[offset:]
        raise KeyError(key)
