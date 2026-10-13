"""Time foundation-model loading and forecasting on 200 toy series (CPU, inside Docker)."""

import time

import numpy as np
import pandas as pd

HORIZON = 28
N_SERIES = 200
rng = np.random.default_rng(0)
t = np.arange(365)
# Weekly pattern plus noise: only meant to check plumbing and speed, not accuracy.
series = [
    (5 + 2 * np.sin(2 * np.pi * t / 7) + rng.poisson(1, size=len(t))).astype("float32")
    for _ in range(N_SERIES)
]


def timed(label, fn):
    start = time.perf_counter()
    result = fn()  # the call finishes before the elapsed time is read
    print(f"  {label}: {time.perf_counter() - start:.1f}s")
    return result


def chronos2() -> None:
    from chronos import Chronos2Pipeline

    pipeline = timed(
        "load", lambda: Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    )
    ds = pd.date_range("2020-01-01", periods=len(t), freq="D")
    context = pd.concat(
        [
            pd.DataFrame({"id": f"s{i}", "timestamp": ds, "target": y})
            for i, y in enumerate(series)
        ],
        ignore_index=True,
    )
    pred = timed(
        f"forecast {N_SERIES} series",
        lambda: pipeline.predict_df(
            context,
            prediction_length=HORIZON,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        ),
    )
    print(f"  output rows: {len(pred)} (expected {N_SERIES * HORIZON})")


def timesfm3() -> None:
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    config = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=32, device="cpu"
    )
    forecaster = timed("load", lambda: TimesFM3Evaluator(config))
    outputs = timed(
        f"forecast {N_SERIES} series",
        lambda: list(
            forecaster.predict_batch(
                series, horizon=HORIZON, return_quantiles=True, use_symmetric_averaging=False
            )
        ),
    )
    print(f"  outputs: {len(outputs)}, quantiles shape {outputs[0].quantiles.shape}")


if __name__ == "__main__":
    for name, fn in [("chronos-2", chronos2), ("timesfm-3.0", timesfm3)]:
        print(name)
        try:
            fn()
        except Exception as exc:  # show the error text instead of a long traceback
            print(f"  FAILED ({type(exc).__name__}: {exc})")
