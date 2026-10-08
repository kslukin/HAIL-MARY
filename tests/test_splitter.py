import pandas as pd
import pytest

from tsagent.backtest.splitter import rolling_origin_windows

H = 7


@pytest.fixture
def df():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    full = pd.DataFrame({"unique_id": "A", "ds": dates, "y": range(100)})
    late = pd.DataFrame({"unique_id": "B", "ds": dates[70:], "y": range(30)})  # starts late
    return pd.concat([full, late], ignore_index=True)


def make_windows(df):
    return list(rolling_origin_windows(df, horizon=H, n_windows=3, min_history=14))


def test_no_look_ahead(df):
    for w in make_windows(df):
        assert w.train["ds"].max() <= w.cutoff
        assert w.test["ds"].min() > w.cutoff
        assert w.test["ds"].max() <= w.cutoff + pd.Timedelta(days=H)


def test_last_window_ends_at_last_date(df):
    assert make_windows(df)[-1].test["ds"].max() == df["ds"].max()


def test_cutoffs_are_spaced_by_horizon(df):
    cutoffs = [w.cutoff for w in make_windows(df)]
    assert all(b - a == pd.Timedelta(days=H) for a, b in zip(cutoffs, cutoffs[1:], strict=False))


def test_every_test_series_has_full_horizon(df):
    for w in make_windows(df):
        assert (w.test.groupby("unique_id").size() == H).all()


def test_min_history_excludes_short_series(df):
    windows = make_windows(df)
    # Series B has only 9 observations before the first cutoff, so it must be excluded there.
    assert set(windows[0].train["unique_id"]) == {"A"}
    assert set(windows[1].train["unique_id"]) == {"A", "B"}
    # Train and test must always describe the same set of series.
    for w in windows:
        assert set(w.train["unique_id"]) == set(w.test["unique_id"])
