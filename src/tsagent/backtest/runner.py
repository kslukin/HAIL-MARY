"""Run a model over all backtest windows and collect the metrics."""

from __future__ import annotations

import pandas as pd

from tsagent.backtest.splitter import rolling_origin_windows
from tsagent.metrics.accuracy import evaluate, summarize
from tsagent.models.base import Forecaster


def run_backtest(
    df: pd.DataFrame,
    model: Forecaster,
    horizon: int = 28,
    n_windows: int = 3,
    freq: str = "D",
    season_length: int = 7,
    min_history: int = 56,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Return per-series metrics for every window: model, window, unique_id, mase, sql, coverage."""
    frames = []
    for w in rolling_origin_windows(df, horizon, n_windows, freq, min_history):
        # The model only receives w.train, i.e. data up to the cutoff.
        forecast = model.forecast(w.train, horizon, list(quantiles))
        scores = evaluate(w.train, w.test, forecast, season_length).reset_index()
        scores.insert(0, "window", w.index)
        scores.insert(0, "model", model.name)
        frames.append(scores)
    return pd.concat(frames, ignore_index=True)


def summarize_by_window(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-series results into one row per (model, window)."""
    rows = []
    for (model, window), g in results.groupby(["model", "window"]):
        summary = summarize(g.set_index("unique_id")[["mase", "sql", "coverage"]])
        rows.append({"model": model, "window": window, **summary.to_dict()})
    return pd.DataFrame(rows)
