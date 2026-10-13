"""Sanity-check the foundation-model wrappers through the common Forecaster interface.

Three series with very different levels: if a wrapper mixes up the series order,
the median forecast of a series will be far from its own level.
"""

import numpy as np
import pandas as pd

from tsagent.models.chronos_model import Chronos2Forecaster
from tsagent.models.timesfm_model import TimesFM3Forecaster

LEVELS = {"low": 1.0, "mid": 50.0, "high": 500.0}
HORIZON = 14
N = 300
ds = pd.date_range("2020-01-01", periods=N, freq="D")
rng = np.random.default_rng(0)
weekly = np.sin(2 * np.pi * np.arange(N) / 7)
history = pd.concat(
    [
        pd.DataFrame(
            {
                "unique_id": uid,
                "ds": ds,
                "y": level * (1 + 0.1 * weekly) + rng.normal(0, 0.02 * level, N),
            }
        )
        for uid, level in LEVELS.items()
    ],
    ignore_index=True,
)


def check(model) -> None:
    fc = model.forecast(history, HORIZON, [0.1, 0.5, 0.9])
    assert list(fc.columns) == ["unique_id", "ds", "q0.1", "q0.5", "q0.9"], list(fc.columns)
    assert len(fc) == len(LEVELS) * HORIZON, len(fc)
    first_day = ds[-1] + pd.Timedelta(days=1)
    assert (fc.groupby("unique_id")["ds"].min() == first_day).all(), "wrong forecast dates"
    assert (fc["q0.1"] <= fc["q0.5"]).all() and (fc["q0.5"] <= fc["q0.9"]).all()
    for uid, level in LEVELS.items():
        median = fc.loc[fc["unique_id"] == uid, "q0.5"].mean()
        assert abs(median / level - 1) < 0.2, f"{uid}: median {median:.1f} vs level {level}"


if __name__ == "__main__":
    for model in [Chronos2Forecaster(), TimesFM3Forecaster()]:
        try:
            check(model)
            print(f"{model.name}: PASS")
        except Exception as exc:  # report the problem instead of a long traceback
            print(f"{model.name}: FAIL ({type(exc).__name__}: {exc})")
