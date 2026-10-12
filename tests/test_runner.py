import pandas as pd

from tsagent.backtest.runner import run_backtest, summarize_by_window
from tsagent.models.seasonal_naive import SeasonalNaive


class SpyModel(SeasonalNaive):
    """Seasonal naive that records the latest date of every history it receives."""

    name = "spy"

    def __init__(self):
        super().__init__()
        self.seen_max_dates = []

    def forecast(self, history, horizon, quantiles):
        self.seen_max_dates.append(history["ds"].max())
        return super().forecast(history, horizon, quantiles)


def make_df():
    ds = pd.date_range("2020-01-01", periods=120, freq="D")
    frames = [
        pd.DataFrame({"unique_id": uid, "ds": ds, "y": [(i % 7) + (i % 3) + k for i in range(120)]})
        for k, uid in enumerate(["A", "B"])
    ]
    return pd.concat(frames, ignore_index=True)


def test_models_never_see_data_after_the_cutoff():
    df = make_df()
    spy = SpyModel()
    results = run_backtest(df, spy, horizon=7, n_windows=3, min_history=14)
    last = df["ds"].max()
    expected_cutoffs = [last - pd.Timedelta(days=7 * (3 - i)) for i in range(3)]
    assert spy.seen_max_dates == expected_cutoffs
    assert set(results["window"]) == {0, 1, 2}


def test_summary_has_one_row_per_window():
    results = run_backtest(make_df(), SeasonalNaive(), horizon=7, n_windows=3, min_history=14)
    summary = summarize_by_window(results)
    assert len(summary) == 3
    assert (summary["n_scored"] == 2).all()
