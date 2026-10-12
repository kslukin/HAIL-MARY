"""Global LightGBM forecaster with direct multi-horizon quantile models."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

LAGS = (0, 1, 2, 3, 4, 5, 6, 13, 20, 27)  # lag k = y[t - k]; lag 0 is the last observed value
MIN_ORIGIN = 27  # earliest origin that still has every lag and a full 28-step window
FEATURES = [f"lag_{k}" for k in LAGS] + [
    "mean_7",
    "mean_28",
    "zero_share_28",
    "snaive",
    "h",
    "dow",
    "month",
]
# Features measured in sales units; they are divided by the series level.
SCALED = [f"lag_{k}" for k in LAGS] + ["mean_7", "mean_28", "snaive"]

DEFAULT_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 50,
    "deterministic": True,  # reproducible results across runs
    "force_row_wise": True,
    "verbose": -1,
}


def build_series_rows(
    y: np.ndarray, dates: pd.DatetimeIndex, origins: np.ndarray, horizon: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Raw (unscaled) feature rows for one series: one row per (origin, horizon step).

    Every feature of a row is computed from y[: origin + 1] only. `dates` must cover
    the history plus `horizon` future steps, so calendar features exist for forecast
    targets. Returns the feature table and, for each row, the index of its target in y
    (valid for training rows; for forecast rows it points past the end of y).
    """
    # Cumulative sums let us get any rolling sum as a difference of two numbers.
    csum = np.concatenate([[0.0], np.cumsum(y)])
    zero_csum = np.concatenate([[0.0], np.cumsum(y == 0)])

    per_origin = {f"lag_{k}": y[origins - k] for k in LAGS}
    per_origin["mean_7"] = (csum[origins + 1] - csum[origins - 6]) / 7
    per_origin["mean_28"] = (csum[origins + 1] - csum[origins - 27]) / 28
    per_origin["zero_share_28"] = (zero_csum[origins + 1] - zero_csum[origins - 27]) / 28

    h = np.tile(np.arange(1, horizon + 1), len(origins))  # horizon step of each row
    target_idx = np.repeat(origins, horizon) + h
    rows = {name: np.repeat(values, horizon) for name, values in per_origin.items()}
    # Same weekday last week: step back whole weeks until the index is <= origin.
    rows["snaive"] = y[target_idx - 7 * np.ceil(h / 7).astype(int)]
    rows["h"] = h
    rows["dow"] = dates.dayofweek.to_numpy()[target_idx]
    rows["month"] = dates.month.to_numpy()[target_idx]
    return pd.DataFrame(rows)[FEATURES], target_idx


def _scale_features(x: pd.DataFrame, scale: float) -> pd.DataFrame:
    x = x.copy()
    x[SCALED] = x[SCALED] / scale
    return x


class LightGBMForecaster:
    """Trains one quantile model per requested quantile on rows from all series."""

    name = "lightgbm"

    def __init__(
        self,
        n_origins: int = 52,
        origin_stride: int = 7,
        freq: str = "D",
        seed: int = 0,
        **lgb_params: object,
    ) -> None:
        self.n_origins = n_origins  # how many training origins per series
        self.origin_stride = origin_stride  # days between consecutive origins
        self.freq = freq
        self.seed = seed
        self.params = {**DEFAULT_PARAMS, **lgb_params}

    def forecast(
        self, history: pd.DataFrame, horizon: int, quantiles: list[float]
    ) -> pd.DataFrame:
        train_x, train_y = [], []
        pred_x, pred_ids, pred_dates, pred_scales = [], [], [], []

        for uid, g in history.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False):
            y = g["y"].to_numpy(dtype=float)
            n = len(y)
            all_dates = pd.date_range(g["ds"].iloc[0], periods=n + horizon, freq=self.freq)
            if all_dates[n - 1] != g["ds"].iloc[-1]:
                raise ValueError(f"series {uid} has gaps; the model needs a regular calendar")

            # Origins walk back from the second-to-last day, so every origin has a target.
            origins = np.arange(n - 2, MIN_ORIGIN - 1, -self.origin_stride)[: self.n_origins]
            if len(origins) == 0:
                raise ValueError(f"series {uid} is too short ({n} observations)")
            scale = max(float(y.mean()), 1e-6)  # series level, from training data only

            x, target_idx = build_series_rows(y, all_dates, origins, horizon)
            keep = target_idx <= n - 1  # a training target must lie inside the history
            train_x.append(_scale_features(x[keep], scale))
            train_y.append(y[target_idx[keep]] / scale)

            # Forecast rows: the origin is the last observed day.
            x_pred, _ = build_series_rows(y, all_dates, np.array([n - 1]), horizon)
            pred_x.append(_scale_features(x_pred, scale))
            pred_ids.append(uid)
            pred_dates.append(pd.Series(all_dates[n:]))
            pred_scales.append(scale)

        x_train = pd.concat(train_x, ignore_index=True)
        y_train = np.concatenate(train_y)
        x_future = pd.concat(pred_x, ignore_index=True)

        qs = sorted(quantiles)
        preds = np.column_stack([self._fit_predict(x_train, y_train, x_future, q) for q in qs])
        preds = np.sort(preds, axis=1)  # quantiles must not cross
        preds = np.maximum(preds * np.repeat(pred_scales, horizon)[:, None], 0.0)  # back to units

        out = pd.DataFrame(
            {
                "unique_id": np.repeat(pred_ids, horizon),
                "ds": pd.concat(pred_dates, ignore_index=True),
            }
        )
        for j, q in enumerate(qs):
            out[f"q{q}"] = preds[:, j]
        return out

    def _fit_predict(
        self, x_train: pd.DataFrame, y_train: np.ndarray, x_future: pd.DataFrame, q: float
    ) -> np.ndarray:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=q, random_state=self.seed, **self.params
        )
        model.fit(x_train, y_train)
        return model.predict(x_future)
