"""
Performance statistics helpers.

Wraps ``pyfolio-reloaded`` and ``scipy.stats`` with proper input handling:

* Bug #13 fix: daily returns are passed as a ``pd.Series`` with a
  ``DatetimeIndex`` so pyfolio annualises correctly.
* Bug #14 fix: uses ``scipy.stats.t.interval(confidence=…)`` (scipy ≥ 1.9).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import scipy.stats as st


def perf_stats_series(
    daily_returns: np.ndarray,
    dates: Sequence,
) -> pd.Series:
    """
    Compute pyfolio performance statistics from a 1-D array of daily returns.

    Parameters
    ----------
    daily_returns : array-like
        Daily percentage returns.
    dates : sequence
        Corresponding dates (must be convertible to ``DatetimeIndex``).

    Returns
    -------
    pd.Series
        Named performance statistics (annual return, Sharpe, etc.).
    """
    import pyfolio.timeseries as ts  # lazy import – heavy dependency

    returns_series = pd.Series(
        daily_returns,
        index=pd.DatetimeIndex(dates),
        name="returns",
    )
    return ts.perf_stats(returns_series)


def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
    axis: int = 0,
):
    """
    Compute a *t*-distribution confidence interval along *axis*.

    Uses the modern ``scipy.stats.t.interval(confidence=…)`` API.
    """
    n = data.shape[axis]
    dof = n - 1
    mean = np.mean(data, axis=axis)
    se = st.sem(data, axis=axis)
    lower, upper = st.t.interval(confidence=confidence, df=dof, loc=mean, scale=se)
    return lower, upper
