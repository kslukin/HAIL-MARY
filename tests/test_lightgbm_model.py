import numpy as np
import pandas as pd
import pytest

from tsagent.backtest.runner import run_backtest
from tsagent.models.lightgbm_model import LightGBMForecaster, build_series_rows


def make_df(n_series=3, n_days=120):
    rng = np.random.default_rng(0)
    ds = pd.date_range("2020-01-01", periods=n_days)
    frames = [
        pd.DataFrame(
            {
                "unique_id": f"S{k}",
                "ds": ds,
                "y": rng.poisson(3 + k, size=n_days).astype(float),
            }
        )
        for k in range(n_series)
    ]
    return pd.concat(frames, ignore_index=True)


def test_features_at_an_origin_ignore_later_observations():
    rng = np.random.default_rng(0)
    y = rng.poisson(3, size=100).astype(float)
    dates = pd.date_range("2020-01-01", periods=107)
    origin = np.array([60])

    base, _ = build_series_rows(y, dates, origin, horizon=7)
    y_changed = y.copy()
    y_changed[61:] += 100.0  # rewrite the entire future after the origin
    changed, _ = build_series_rows(y_changed, dates, origin, horizon=7)

    pd.testing.assert_frame_equal(base, changed)


def test_target_index_is_origin_plus_horizon():
    y = np.arange(60, dtype=float)
    dates = pd.date_range("2020-01-01", periods=67)
    x, target_idx = build_series_rows(y, dates, np.array([40]), horizon=7)
    assert target_idx.tolist() == [41, 42, 43, 44, 45, 46, 47]
    assert x["h"].tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_snaive_feature_is_the_same_weekday_last_week():
    y = np.arange(60, dtype=float)  # y[i] == i, so the feature equals the index it points to
    dates = pd.date_range("2020-01-01", periods=74)
    x, _ = build_series_rows(y, dates, np.array([40]), horizon=14)
    assert x["snaive"].tolist() == [34, 35, 36, 37, 38, 39, 40] * 2


def test_forecast_has_expected_shape_and_valid_quantiles():
    df = make_df()
    model = LightGBMForecaster(n_estimators=20, min_child_samples=5)
    fc = model.forecast(df, horizon=14, quantiles=[0.1, 0.5, 0.9])
    assert list(fc.columns) == ["unique_id", "ds", "q0.1", "q0.5", "q0.9"]
    assert len(fc) == 3 * 14
    assert fc["ds"].min() == df["ds"].max() + pd.Timedelta(days=1)
    assert (fc["q0.1"] <= fc["q0.5"]).all()
    assert (fc["q0.5"] <= fc["q0.9"]).all()
    assert (fc[["q0.1", "q0.5", "q0.9"]] >= 0).all().all()


def test_series_with_gaps_is_rejected():
    df = make_df().drop(index=10)  # removes one day from the middle of the first series
    with pytest.raises(ValueError, match="gaps"):
        LightGBMForecaster(n_estimators=5).forecast(df, 7, [0.5])


def test_model_runs_inside_the_backtest():
    model = LightGBMForecaster(n_estimators=10, min_child_samples=5)
    results = run_backtest(make_df(), model, horizon=7, n_windows=2, min_history=56)
    assert set(results["model"]) == {"lightgbm"}
    assert len(results) == 3 * 2  # 3 series x 2 windows
    assert results["mase"].notna().all()
