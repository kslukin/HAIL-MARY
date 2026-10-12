"""Run the seasonal naive baseline on an M5 subset and print metrics per window."""

import pandas as pd

from tsagent.backtest.runner import run_backtest, summarize_by_window
from tsagent.data.m5 import load_m5
from tsagent.models.seasonal_naive import SeasonalNaive


def main() -> None:
    long, _ = load_m5("data/raw/m5", n_series=200, seed=0)
    results = run_backtest(long, SeasonalNaive())
    pd.set_option("display.width", 120)
    print(summarize_by_window(results).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
