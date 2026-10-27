"""Does picking a model per series from past backtest results beat always using one model?"""

from __future__ import annotations

import numpy as np
import pandas as pd


def selection_vs_fixed(mase: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
    """Per-series MASE of fixed policies and of 'pick the best model of the past windows'.

    For window k the model is chosen from the mean MASE over up to `lookback` windows
    before k. Window k+1 starts exactly where the test period of window k ends, so every
    actual used for the choice was already known at the cutoff of window k+1: no look-ahead.
    The hindsight column (best model picked after seeing the window's own result) cannot be
    achieved in practice, and it is biased downwards: the minimum of several noisy numbers
    is low even when the models are equally good.

    Args:
        mase: rows (window, unique_id), one column per candidate model, no missing values.
        lookback: how many previous windows are averaged before choosing a model.
    """
    window_ids = mase.index.get_level_values("window")
    windows = sorted(window_ids.unique())
    frames = []
    for j in range(1, len(windows)):
        curr = windows[j]
        past = windows[max(0, j - lookback) : j]
        # Mean MASE over the past windows, per series and model.
        before = mase[window_ids.isin(past)].groupby(level="unique_id").mean()
        after = mase.xs(curr, level="window")
        common = before.index.intersection(after.index)  # series with history and a current score
        before, after = before.loc[common], after.loc[common]

        chosen = before.idxmin(axis=1)  # model with the lowest past MASE
        columns = after.columns.get_indexer(chosen.to_numpy())
        frame = after.copy()
        frame["selected"] = after.to_numpy()[np.arange(len(after)), columns]
        frame["hindsight_best"] = after.min(axis=1)
        frame["chosen_model"] = chosen
        frame["window"] = curr
        frames.append(frame.reset_index())
    return pd.concat(frames, ignore_index=True)
