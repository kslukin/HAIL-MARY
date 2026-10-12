"""Common interface that every forecasting model implements."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Forecaster(Protocol):
    """A model that turns a history into quantile forecasts.

    The backtest and, later, the agent only rely on this interface, so a new
    model (LightGBM, TimesFM, Chronos-2) plugs in without changing anything else.
    """

    name: str

    def forecast(
        self,
        history: pd.DataFrame,  # columns: unique_id, ds, y (data up to the cutoff only)
        horizon: int,
        quantiles: list[float],
    ) -> pd.DataFrame:
        """Return unique_id, ds and one column per quantile, e.g. q0.1, q0.5, q0.9."""
        ...
