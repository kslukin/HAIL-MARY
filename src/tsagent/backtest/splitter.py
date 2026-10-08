"""Rolling-origin backtest splits without look-ahead."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd
from pandas.tseries.frequencies import to_offset


@dataclass(frozen=True)
class Window:
    """One backtest fold: models may only look at `train`; `test` is scored."""

    index: int
    cutoff: pd.Timestamp  # last timestamp that models are allowed to see
    train: pd.DataFrame  # rows with ds <= cutoff
    test: pd.DataFrame  # the `horizon` steps right after the cutoff


def rolling_origin_windows(
    df: pd.DataFrame,
    horizon: int = 28,
    n_windows: int = 3,
    freq: str = "D",
    min_history: int = 56,
) -> Iterator[Window]:
    """Yield chronological rolling-origin windows over a long dataframe.

    The last window's test period ends at the last date in `df`; each earlier
    window is shifted back by `horizon` steps. Cutoffs are global dates (all
    series share one calendar in M5), not per-series positions.

    Args:
        df: Long dataframe with columns unique_id, ds, y.
        horizon: Forecast length in steps of `freq`.
        n_windows: Number of backtest windows.
        freq: Pandas frequency alias of the data ("D" for daily, "h" for hourly).
        min_history: A series enters a window only if it has at least this many
            observations up to the cutoff. Without it, newly launched items
            would be forecast from almost no history and distort the metrics.
    """
    step = to_offset(freq) * horizon  # one horizon expressed as a time offset
    last = df["ds"].max()

    for i in range(n_windows):
        # Oldest window first: i=0 has the earliest cutoff.
        cutoff = last - step * (n_windows - i)
        end = cutoff + step

        train = df[df["ds"] <= cutoff]  # strictly no data after the cutoff
        counts = train.groupby("unique_id").size()
        eligible = counts[counts >= min_history].index
        train = train[train["unique_id"].isin(eligible)]

        in_test = (df["ds"] > cutoff) & (df["ds"] <= end)
        test = df[in_test & df["unique_id"].isin(eligible)]

        yield Window(i, cutoff, train.reset_index(drop=True), test.reset_index(drop=True))
