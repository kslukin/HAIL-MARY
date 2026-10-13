"""Common interface that every forecasting model implements."""

from __future__ import annotations

from typing import Protocol

import numpy as np
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


def finalize_quantiles(frame: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    """Sort quantile columns within each row and clip at zero (sales cannot be negative).

    Applied identically to every model, so that no model is helped or hurt by crossing
    quantiles or negative values.
    """
    cols = [f"q{q}" for q in sorted(quantiles)]
    values = np.sort(frame[cols].to_numpy(dtype=float), axis=1)
    frame[cols] = np.maximum(values, 0.0)
    return frame
