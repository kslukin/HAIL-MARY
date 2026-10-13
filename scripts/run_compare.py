"""Compare forecasting models on an M5 subset with identical backtest windows."""

import argparse
import time
from pathlib import Path

import pandas as pd

from tsagent.backtest.runner import run_backtest, summarize_by_window
from tsagent.data.m5 import load_m5
from tsagent.models.chronos_model import Chronos2Forecaster
from tsagent.models.lightgbm_model import LightGBMForecaster
from tsagent.models.seasonal_naive import SeasonalNaive
from tsagent.models.timesfm_model import TimesFM3Forecaster

# Classes, not instances: a model is only built (and PyTorch only loaded) when requested.
MODELS = {
    "seasonal_naive": SeasonalNaive,
    "lightgbm": LightGBMForecaster,
    "chronos2": Chronos2Forecaster,
    "timesfm3": TimesFM3Forecaster,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--n-series", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/m5_subset")
    args = parser.parse_args()

    long, _ = load_m5("data/raw/m5", n_series=args.n_series, seed=args.seed)

    frames = []
    for name in args.models:
        start = time.perf_counter()
        frames.append(run_backtest(long, MODELS[name]()))  # finishes before the time is read
        print(f"{name}: {time.perf_counter() - start:.0f}s for all windows")
    results = pd.concat(frames, ignore_index=True)

    pd.set_option("display.width", 140)
    summary = summarize_by_window(results)
    print("\nPer window:")
    print(summary.round(3).to_string(index=False))

    print("\nAverage over windows:")
    columns = ["mase", "mase_median", "sql", "coverage", "n_scored"]
    print(summary.groupby("model")[columns].mean().round(3).to_string())

    # Same series and windows for every model, so paired comparisons are fair.
    paired = results.pivot_table(index=["window", "unique_id"], columns="model", values="mase")
    if "seasonal_naive" in paired:
        for model in paired.columns.drop("seasonal_naive"):
            pair = paired[[model, "seasonal_naive"]].dropna()
            wins = (pair[model] < pair["seasonal_naive"]).mean()
            print(f"{model} beats seasonal_naive on {wins:.0%} of (window, series) pairs")
    best = paired.dropna().idxmin(axis=1).value_counts(normalize=True)
    print("\nShare of (window, series) pairs where each model has the lowest MASE:")
    print(best.round(3).to_string())

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    results.to_csv(f"{out}_per_series.csv", index=False)
    summary.to_csv(f"{out}_summary.csv", index=False)


if __name__ == "__main__":
    main()
