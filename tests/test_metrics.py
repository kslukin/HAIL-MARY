import pandas as pd
import pytest

from tsagent.metrics.accuracy import evaluate, summarize


def make_inputs(train_values, test_values, quantiles):
    train_ds = pd.date_range("2020-01-01", periods=len(train_values))
    test_ds = pd.date_range(train_ds[-1] + pd.Timedelta(days=1), periods=len(test_values))
    train = pd.DataFrame({"unique_id": "A", "ds": train_ds, "y": train_values})
    test = pd.DataFrame({"unique_id": "A", "ds": test_ds, "y": test_values})
    forecast = pd.DataFrame({"unique_id": "A", "ds": test_ds, **quantiles})
    return train, test, forecast


TRAIN = list(range(1, 11))  # consecutive values differ by 1, so the naive scale is exactly 1


def test_single_point_metrics_match_hand_calculation():
    # |10 - 9| / 1 = 1; pinball: q0.1 -> 0.2, q0.5 -> 0.5, q0.9 -> 0.2, mean 0.3; 10 is in [8, 12]
    train, test, fc = make_inputs(TRAIN, [10], {"q0.1": [8], "q0.5": [9], "q0.9": [12]})
    row = evaluate(train, test, fc, season_length=1).loc["A"]
    assert row["mase"] == pytest.approx(1.0)
    assert row["sql"] == pytest.approx(0.3)
    assert row["coverage"] == pytest.approx(1.0)


def test_mase_averages_absolute_errors_over_the_horizon():
    # errors |11-10| = 1 and |12-14| = 2 -> mean 1.5, scale 1 -> MASE 1.5
    train, test, fc = make_inputs(
        TRAIN,
        [11, 12],
        {"q0.1": [9, 13], "q0.5": [10, 14], "q0.9": [11, 15]},
    )
    assert evaluate(train, test, fc, season_length=1).loc["A", "mase"] == pytest.approx(1.5)


def test_coverage_is_zero_when_actual_is_outside_the_interval():
    train, test, fc = make_inputs(TRAIN, [20], {"q0.1": [8], "q0.5": [9], "q0.9": [12]})
    assert evaluate(train, test, fc, season_length=1).loc["A", "coverage"] == 0.0


def test_zero_scale_series_is_nan_and_excluded_from_summary():
    dates = pd.date_range("2020-01-01", periods=10)
    train = pd.concat(
        [
            pd.DataFrame({"unique_id": "A", "ds": dates, "y": TRAIN}),
            pd.DataFrame({"unique_id": "B", "ds": dates, "y": [5] * 10}),  # constant -> scale 0
        ]
    )
    day = pd.Timestamp("2020-01-11")
    test = pd.DataFrame({"unique_id": ["A", "B"], "ds": day, "y": [11, 5]})
    fc = pd.DataFrame(
        {
            "unique_id": ["A", "B"],
            "ds": day,
            "q0.1": [10, 5],
            "q0.5": [11, 5],
            "q0.9": [12, 5],
        }
    )
    per_series = evaluate(train, test, fc, season_length=1)
    assert pd.isna(per_series.loc["B", "mase"])
    summary = summarize(per_series)
    assert summary["n_series"] == 2
    assert summary["n_scored"] == 1
    assert summary["mase"] == pytest.approx(0.0)


def test_missing_forecast_rows_raise():
    train, test, fc = make_inputs(
        TRAIN, [11, 12], {"q0.1": [9, 9], "q0.5": [10, 10], "q0.9": [11, 11]}
    )
    with pytest.raises(ValueError, match="missing"):
        evaluate(train, test, fc.iloc[:1], season_length=1)  # forecast for day 2 is absent
