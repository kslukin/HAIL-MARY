import pandas as pd
import pytest

from tsagent.data.m5 import load_m5

DAYS = [f"d_{i}" for i in range(1, 6)]
SERIES = {
    "A_1_CA_1": [0, 0, 3, 0, 2],  # starts selling on day 3
    "B_1_CA_1": [1, 1, 1, 1, 1],
    "C_1_CA_1": [0, 0, 0, 0, 0],  # never sold
}


@pytest.fixture
def raw_dir(tmp_path):
    calendar = pd.DataFrame(
        {"d": DAYS, "date": pd.date_range("2011-01-29", periods=5).strftime("%Y-%m-%d")}
    )
    calendar.to_csv(tmp_path / "calendar.csv", index=False)

    sales = pd.DataFrame(
        {
            "id": [f"{uid}_evaluation" for uid in SERIES],
            "item_id": [f"ITEM_{uid[0]}" for uid in SERIES],
            "dept_id": "DEPT_1",
            "cat_id": "CAT",
            "store_id": "CA_1",
            "state_id": "CA",
        }
    )
    for i, day in enumerate(DAYS):
        sales[day] = [values[i] for values in SERIES.values()]
    sales.to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    return tmp_path


def test_long_format_without_trimming(raw_dir):
    long, meta = load_m5(raw_dir, trim_leading_zeros=False)
    assert list(long.columns) == ["unique_id", "ds", "y"]
    assert len(long) == 15
    assert set(long["unique_id"]) == set(SERIES)  # "_evaluation" suffix removed
    assert long["ds"].min() == pd.Timestamp("2011-01-29")
    assert len(meta) == 3


def test_trim_leading_zeros_and_drop_all_zero_series(raw_dir):
    long, meta = load_m5(raw_dir, trim_leading_zeros=True)
    a = long[long["unique_id"] == "A_1_CA_1"]
    assert len(a) == 3
    assert a["ds"].min() == pd.Timestamp("2011-01-31")
    assert "C_1_CA_1" not in set(long["unique_id"])
    assert "C_1_CA_1" not in set(meta["unique_id"])
    assert len(long) == 8


def test_sampling_is_reproducible(raw_dir):
    first, _ = load_m5(raw_dir, n_series=2, seed=1, trim_leading_zeros=False)
    second, _ = load_m5(raw_dir, n_series=2, seed=1, trim_leading_zeros=False)
    assert first["unique_id"].nunique() == 2
    pd.testing.assert_frame_equal(first, second)
