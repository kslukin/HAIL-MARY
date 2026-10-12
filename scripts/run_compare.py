"""Compare seasonal naive and LightGBM on an M5 subset with the same backtest windows."""

from pathlib import Path

import pandas as pd

from tsagent.backtest.runner import run_backtest, summarize_by_window
from tsagent.data.m5 import load_m5
from tsagent.models.lightgbm_model import LightGBMForecaster
from tsagent.models.seasonal_naive import SeasonalNaive


def main() -> None:
    long, _ = load_m5("data/raw/m5", n_series=200, seed=0)
    models = [SeasonalNaive(), LightGBMForecaster()]
    results = pd.concat([run_backtest(long, m) for m in models], ignore_index=True)

    pd.set_option("display.width", 120)
    summary = summarize_by_window(results)
    print(summary.round(3).to_string(index=False))

    print("\nAverage over windows:")
    columns = ["mase", "mase_median", "sql", "coverage"]
    print(summary.groupby("model")[columns].mean().round(3).to_string())

    # Same series and windows for both models, so a paired comparison is fair.
    paired = results.pivot_table(index=["window", "unique_id"], columns="model", values="mase")
    wins = (paired["lightgbm"] < paired["seasonal_naive"]).mean()
    print(f"\nLightGBM beats seasonal naive on {wins:.0%} of (window, series) pairs")

    # Save the numbers so the table does not have to be recomputed (small files, committed).
    Path("results").mkdir(exist_ok=True)
    results.to_csv("results/m5_subset_per_series.csv", index=False)
    summary.to_csv("results/m5_subset_summary.csv", index=False)


if __name__ == "__main__":
    main()
