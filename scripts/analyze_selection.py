"""Evaluate model selection from past backtest windows against fixed single-model policies."""

import argparse

import pandas as pd

from tsagent.analysis.compare import bootstrap_mean_ci, pivot_mase
from tsagent.analysis.selection import selection_vs_fixed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="results/m5_n1000")
    parser.add_argument("--lookbacks", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()

    results = pd.read_csv(f"{args.prefix}_per_series.csv")
    mase = pivot_mase(results)
    models = list(mase.columns)
    n_windows = mase.index.get_level_values("window").nunique()
    print(f"{n_windows} windows; selection is evaluated on windows 1..{n_windows - 1}")

    for lookback in args.lookbacks:
        policies = selection_vs_fixed(mase, lookback)
        fixed = policies[models].mean()
        # The best fixed model is picked on these same data, which favours the fixed policy.
        best_fixed = fixed.idxmin()
        diff = (policies["selected"] - policies[best_fixed]).groupby(policies["unique_id"]).mean()
        mean, low, high = bootstrap_mean_ci(diff.to_numpy())

        print(f"\nlookback = {lookback} window(s), {len(policies)} (window, series) pairs")
        print(f"  selected MASE       {policies['selected'].mean():.3f}")
        print(f"  best fixed ({best_fixed})  {fixed[best_fixed]:.3f}")
        print(f"  hindsight (biased)  {policies['hindsight_best'].mean():.3f}")
        print(f"  selected - best fixed: {mean:+.4f} (95% CI {low:+.4f} to {high:+.4f})")
        shares = policies["chosen_model"].value_counts(normalize=True).round(3)
        print("  share of choices:", ", ".join(f"{m} {s}" for m, s in shares.items()))
        policies.to_csv(f"{args.prefix}_selection_lb{lookback}.csv", index=False)

    print("\nMean MASE of fixed policies (same pairs for every lookback):")
    print(fixed.sort_values().round(3).to_string())


if __name__ == "__main__":
    main()
