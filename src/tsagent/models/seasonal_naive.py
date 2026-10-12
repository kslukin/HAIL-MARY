"""Seasonal naive baseline: repeat the last observed season."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalNaive:
    """Forecast = the last `season_length` observations, repeated cyclically.

    Intervals come from the model's own in-sample errors (y_t - y_{t-season}):
    the lower quantiles use the negative part of the error quantile, the upper
    ones the positive part. This keeps q_low <= median <= q_high and forecasts
    are clipped at zero, because sales cannot be negative.
    """

    name = "seasonal_naive"

    def __init__(self, season_length: int = 7, freq: str = "D") -> None:
        self.season_length = season_length
        self.freq = freq

    def forecast(
        self, history: pd.DataFrame, horizon: int, quantiles: list[float]
    ) -> pd.DataFrame:
        m = self.season_length
        frames = []
        for uid, g in history.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False):
            y = g["y"].to_numpy(dtype=float)
            if len(y) <= m:
                raise ValueError(f"series {uid} has too little history for season length {m}")

            # np.resize repeats the array cyclically: last week, last week, ...
            point = np.resize(y[-m:], horizon)
            # How wrong "same day last week" was on the training history.
            errors = y[m:] - y[:-m]

            frame = pd.DataFrame(
                {
                    "unique_id": uid,
                    "ds": pd.date_range(g["ds"].iloc[-1], periods=horizon + 1, freq=self.freq)[1:],
                }
            )
            for q in quantiles:
                if q == 0.5:
                    values = point
                else:
                    offset = float(np.quantile(errors, q))
                    # Never let a lower quantile sit above the median or an upper one below it.
                    offset = min(offset, 0.0) if q < 0.5 else max(offset, 0.0)
                    values = np.maximum(point + offset, 0.0)
                frame[f"q{q}"] = values
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
