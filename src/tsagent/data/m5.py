"""Load the M5 (Walmart) dataset into the long format used across the project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SALES_FILE = "sales_train_evaluation.csv"
META_COLS = ["unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_m5(
    raw_dir: str | Path,
    n_series: int | None = None,
    seed: int = 0,
    trim_leading_zeros: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load M5 sales as a long dataframe plus a per-series metadata table.

    Args:
        raw_dir: Directory with calendar.csv and sales_train_evaluation.csv.
        n_series: If set, randomly sample this many series BEFORE reshaping.
            The full dataset has ~30k series x 1941 days (~59M rows in long format),
            so sampling first keeps memory low.
        seed: Random seed for the sample (makes subsets reproducible).
        trim_leading_zeros: Drop days before a series' first non-zero sale
            (the item was most likely not on sale yet). Series that are zero
            everywhere are dropped entirely.

    Returns:
        long: columns unique_id, ds (datetime), y (units sold), sorted by id and date.
        meta: one row per series with item/dept/cat/store/state ids.
    """
    raw_dir = Path(raw_dir)
    calendar = pd.read_csv(raw_dir / "calendar.csv", usecols=["d", "date"], parse_dates=["date"])

    sales_path = raw_dir / SALES_FILE
    header = pd.read_csv(sales_path, nrows=0).columns
    day_cols = [c for c in header if c.startswith("d_")]
    wide = pd.read_csv(sales_path, dtype={c: "int32" for c in day_cols})

    if n_series is not None:
        wide = wide.sample(n=n_series, random_state=seed)
    wide = wide.sort_values("id").reset_index(drop=True)
    wide["id"] = wide["id"].str.removesuffix("_evaluation")
    wide = wide.rename(columns={"id": "unique_id"})

    meta = wide[META_COLS].copy()

    long = wide.melt(id_vars="unique_id", value_vars=day_cols, var_name="d", value_name="y")
    long = long.merge(calendar, on="d", how="left").rename(columns={"date": "ds"})
    if long["ds"].isna().any():
        raise ValueError("Some day columns have no matching date in calendar.csv")
    long = long[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)

    if trim_leading_zeros:
        started = (long["y"] > 0).astype("int8").groupby(long["unique_id"]).cummax().astype(bool)
        long = long[started].reset_index(drop=True)
        kept = long["unique_id"].unique()
        meta = meta[meta["unique_id"].isin(kept)].reset_index(drop=True)

    return long, meta
