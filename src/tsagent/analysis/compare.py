"""Statistical comparison of models and a per-segment breakdown of backtest results."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from tsagent.backtest.splitter import rolling_origin_windows


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of `values` and its 95% percentile bootstrap interval (resampling the items)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def pivot_mase(results: pd.DataFrame) -> pd.DataFrame:
    """Rows are (window, unique_id), one MASE column per model; only fully scored rows stay."""
    table = results.pivot_table(index=["window", "unique_id"], columns="model", values="mase")
    return table.dropna()


def paired_comparison(mase: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """For every model pair: mean MASE difference A - B with a bootstrap interval.

    A negative difference means A has the lower (better) MASE. The series is the unit
    that gets resampled, because the windows of one series are not independent.
    """
    rows = []
    for a, b in combinations(mase.columns, 2):
        # One number per series: the difference averaged over its windows.
        diff = (mase[a] - mase[b]).groupby("unique_id").mean().to_numpy()
        mean, low, high = bootstrap_mean_ci(diff, n_boot, seed)
        rows.append(
            {
                "A": a,
                "B": b,
                "mean_A_minus_B": mean,
                "ci_low": low,
                "ci_high": high,
                "median_A_minus_B": float(np.median(diff)),
                "A_lower_share": float((diff < 0).mean()),
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        )
    return pd.DataFrame(rows)


def segment_features(
    long: pd.DataFrame,
    horizon: int = 28,
    n_windows: int = 3,
    freq: str = "D",
    min_history: int = 56,
    recent: int = 56,
) -> pd.DataFrame:
    """Series type at each window's cutoff, computed from that window's training data only."""
    frames = []
    for w in rolling_origin_windows(long, horizon, n_windows, freq, min_history):
        train = w.train.sort_values(["unique_id", "ds"])
        tail = train.groupby("unique_id").tail(recent)  # the most recent `recent` days
        feats = pd.DataFrame(
            {
                "n_obs": train.groupby("unique_id").size(),
                "zero_share": tail["y"].eq(0).groupby(tail["unique_id"]).mean(),
                "level": tail["y"].groupby(tail["unique_id"]).mean(),
            }
        )
        feats["window"] = w.index
        frames.append(feats.reset_index())
    feats = pd.concat(frames, ignore_index=True).set_index(["window", "unique_id"])

    # Thresholds are a judgement call, so they are stated in the labels themselves.
    feats["length"] = np.where(feats["n_obs"] < 365, "short (<365 days)", "long (>=365 days)")
    feats["intermittency"] = np.where(
        feats["zero_share"] >= 0.5, "intermittent (>=50% zero days)", "regular"
    )
    # Ranking first makes the three volume groups equal in size even with many ties.
    feats["volume"] = pd.qcut(feats["level"].rank(method="first"), 3, labels=["low", "mid", "high"])
    return feats


def segment_tables(mase: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Mean MASE per model inside every segment, with row counts and the best model."""
    dims = ["length", "intermittency", "volume"]
    data = mase.join(feats[dims], how="inner")
    models = list(mase.columns)
    tables = []
    for dim in dims:
        grouped = data.groupby(dim, observed=True)
        table = grouped[models].mean()
        table.insert(0, "n", grouped.size())
        table["best_model"] = table[models].idxmin(axis=1)
        table.insert(0, "dimension", dim)
        tables.append(table.reset_index().rename(columns={dim: "segment"}))
    return pd.concat(tables, ignore_index=True)
