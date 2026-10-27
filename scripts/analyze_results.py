"""Confidence intervals and segment analysis for a saved model comparison."""

import argparse

import pandas as pd

from tsagent.analysis.compare import (
    paired_comparison,
    pivot_mase,
    segment_features,
    segment_tables,
)
from tsagent.data.m5 import load_m5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="results/m5_n1000")
    parser.add_argument("--n-series", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-windows", type=int, default=3)
    args = parser.parse_args()

    results = pd.read_csv(f"{args.prefix}_per_series.csv")
    mase = pivot_mase(results)
    if mase.index.get_level_values("window").nunique() != args.n_windows:
        raise ValueError("--n-windows does not match the comparison run")
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 20)
    print(f"{len(mase)} (window, series) pairs scored by all of: {list(mase.columns)}")

    ci = paired_comparison(mase)
    print("\nPaired MASE difference A - B (negative = A is better), 95% bootstrap over series:")
    print(ci.round(3).to_string(index=False))

    # The same sample as in the comparison run: same --n-series and --seed.
    long, _ = load_m5("data/raw/m5", n_series=args.n_series, seed=args.seed)
    if set(results["unique_id"]) - set(long["unique_id"]):
        raise ValueError("results do not match --n-series / --seed of the comparison run")
    tables = segment_tables(mase, segment_features(long, n_windows=args.n_windows))
    print("\nMean MASE by segment (n counts window-series pairs; lower is better):")
    print(tables.round(3).to_string(index=False))

    ci.to_csv(f"{args.prefix}_ci.csv", index=False)
    tables.to_csv(f"{args.prefix}_segments.csv", index=False)


if __name__ == "__main__":
    main()
