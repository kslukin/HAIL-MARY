"""Forecast accuracy metrics: MASE, scaled quantile loss and interval coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_columns(forecast: pd.DataFrame) -> dict[float, str]:
    """Map quantile level -> column name for columns named like 'q0.1', 'q0.5', 'q0.9'."""
    found: dict[float, str] = {}
    for col in forecast.columns:
        if col.startswith("q"):
            try:
                found[float(col[1:])] = col
            except ValueError:
                continue  # a column that merely starts with "q", not a quantile
    return dict(sorted(found.items()))


def seasonal_naive_scale(train: pd.DataFrame, season_length: int = 7) -> pd.Series:
    """Mean absolute in-sample error of the seasonal naive forecast, per series.
    This is the denominator of MASE. Series with zero scale (the seasonal naive
    forecast was perfect in-sample, typical for all-zero or constant series) get
    NaN, because dividing by zero would make MASE meaningless."""

    def _scale(y: pd.Series) -> float:
        values = y.to_numpy(dtype=float)
        if len(values) <= season_length:
            return np.nan  # not enough history to form a single seasonal difference
        return float(np.abs(values[season_length:] - values[:-season_length]).mean())

    ordered = train.sort_values(["unique_id", "ds"])
    scale = ordered.groupby("unique_id")["y"].apply(_scale)
    return scale.where(scale > 0).rename("scale")  # zero scale -> NaN


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    forecast: pd.DataFrame,
    season_length: int = 7,
) -> pd.DataFrame:
    """Score one backtest window and return one row of metrics per series.

    Args:
        train: Long dataframe (unique_id, ds, y) that the model was allowed to see.
        test: Long dataframe (unique_id, ds, y) with the actual future values.
        forecast: unique_id, ds and quantile columns such as q0.1, q0.5, q0.9.
            The median q0.5 is used as the point forecast, which is the right
            choice for MASE because absolute error is minimised by the median.
        season_length: Seasonal period of the data (7 for daily, 24 for hourly).

    Returns:
        DataFrame indexed by unique_id with columns mase, sql (scaled quantile
        loss) and coverage (share of test points inside the lowest-highest
        quantile interval)."""
    
    qs = quantile_columns(forecast)
    if 0.5 not in qs or min(qs) >= 0.5 or max(qs) <= 0.5:
        raise ValueError("forecast needs q0.5 plus at least one lower and one upper quantile")

    # Left join: every test point must have a forecast, otherwise the score would be flattered.
    merged = test.merge(forecast, on=["unique_id", "ds"], how="left")
    if merged[list(qs.values())].isna().any().any():
        raise ValueError("forecast is missing for some test rows")

    merged["abs_err"] = (merged["y"] - merged[qs[0.5]]).abs()

    # Pinball loss for each quantile: under-forecasts are penalised by q, over-forecasts by 1-q.
    pinball_cols = []
    for q, col in qs.items():
        diff = merged["y"] - merged[col]
        name = f"_pinball_{q}"
        merged[name] = np.maximum(q * diff, (q - 1) * diff)
        pinball_cols.append(name)
    merged["pinball"] = merged[pinball_cols].mean(axis=1)

    lower, upper = qs[min(qs)], qs[max(qs)]
    merged["coverage"] = ((merged["y"] >= merged[lower]) & (merged["y"] <= merged[upper])).astype(
        float
    )

    per_series = merged.groupby("unique_id")[["abs_err", "pinball", "coverage"]].mean()
    # Scale uses training data only, so no information from the test period leaks in.
    per_series = per_series.join(seasonal_naive_scale(train, season_length))
    per_series["mase"] = per_series["abs_err"] / per_series["scale"]
    per_series["sql"] = per_series["pinball"] / per_series["scale"]
    return per_series[["mase", "sql", "coverage"]]


def summarize(per_series: pd.DataFrame) -> pd.Series:
    """Aggregate per-series metrics, excluding series whose scale is undefined.

    n_series vs n_scored shows how many series had to be left out (zero scale),
    so the exclusion is always visible next to the numbers.
    """
    scored = per_series.dropna(subset=["mase"])
    return pd.Series(
        {
            "mase": scored["mase"].mean(),
            "mase_median": scored["mase"].median(),  # robust to a few extreme series
            "sql": scored["sql"].mean(),
            "coverage": scored["coverage"].mean(),
            "n_series": len(per_series),
            "n_scored": len(scored),
        }
    )
