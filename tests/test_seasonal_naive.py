import numpy as np
import pandas as pd
import pytest

from tsagent.models.seasonal_naive import SeasonalNaive


def make_history(values, uid="A"):
    ds = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"unique_id": uid, "ds": ds, "y": values})


def test_point_forecast_repeats_last_cycle():
    # The last week is 1..7, so a 10-day forecast must be 1..7 and then 1, 2, 3.
    history = make_history([0] * 7 + [1, 2, 3, 4, 5, 6, 7])
    fc = SeasonalNaive().forecast(history, horizon=10, quantiles=[0.1, 0.5, 0.9])
    assert fc["q0.5"].tolist() == [1, 2, 3, 4, 5, 6, 7, 1, 2, 3]


def test_forecast_dates_start_right_after_the_last_observation():
    fc = SeasonalNaive().forecast(make_history(list(range(14))), 3, [0.5])
    assert fc["ds"].tolist() == list(pd.date_range("2020-01-15", periods=3, freq="D"))


def test_quantiles_are_ordered_and_non_negative():
    rng = np.random.default_rng(0)
    history = make_history(rng.poisson(2, size=60).tolist())
    fc = SeasonalNaive().forecast(history, 14, [0.1, 0.5, 0.9])
    assert (fc["q0.1"] <= fc["q0.5"]).all()
    assert (fc["q0.5"] <= fc["q0.9"]).all()
    assert (fc[["q0.1", "q0.5", "q0.9"]] >= 0).all().all()


def test_interval_offsets_come_from_in_sample_errors():
    # Weekly blocks of 0, 2, 4, 6: every in-sample error y_t - y_{t-7} equals +2.
    # Point forecast is 6; the upper offset is +2, the lower offset is capped at 0.
    fc = SeasonalNaive().forecast(make_history([i // 7 * 2 for i in range(28)]), 3, [0.1, 0.5, 0.9])
    assert fc["q0.5"].tolist() == [6.0] * 3
    assert fc["q0.9"].tolist() == [8.0] * 3
    assert fc["q0.1"].tolist() == [6.0] * 3


def test_too_short_history_raises():
    with pytest.raises(ValueError):
        SeasonalNaive().forecast(make_history([1, 2, 3]), 5, [0.5])
