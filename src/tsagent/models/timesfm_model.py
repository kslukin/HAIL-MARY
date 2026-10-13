"""Zero-shot TimesFM 3.0 (Google) behind the common Forecaster interface."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsagent.models.base import finalize_quantiles


class TimesFM3Forecaster:
    name = "timesfm3"

    def __init__(
        self,
        checkpoint: str = "google/timesfm-3.0-pytorch",
        device: str = "cpu",
        batch_size: int = 32,
        freq: str = "D",
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.batch_size = batch_size
        self.freq = freq
        self._model = None  # loaded on first use, so importing this module needs no torch

    def _load(self):
        if self._model is None:
            from timesfm3 import ModelConfig, TimesFM3Evaluator

            config = ModelConfig(
                checkpoint_path=self.checkpoint,
                per_core_batch_size=self.batch_size,
                device=self.device,
            )
            self._model = TimesFM3Evaluator(config)
        return self._model

    @staticmethod
    def decile_index(q: float) -> int:
        """Column of a quantile level in the 9 output deciles (0.1 -> 0, 0.5 -> 4, 0.9 -> 8)."""
        scaled = q * 10
        index = round(scaled) - 1
        if abs(scaled - round(scaled)) > 1e-9 or not 0 <= index <= 8:
            raise ValueError(f"TimesFM returns deciles 0.1 ... 0.9 only, got {q}")
        return index

    def forecast(
        self, history: pd.DataFrame, horizon: int, quantiles: list[float]
    ) -> pd.DataFrame:
        qs = sorted(quantiles)
        columns = [self.decile_index(q) for q in qs]

        ids, contexts, last_dates = [], [], []
        for uid, g in history.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=False):
            ids.append(uid)
            contexts.append(g["y"].to_numpy(dtype=np.float32))
            last_dates.append(g["ds"].iloc[-1])

        outputs = list(
            self._load().predict_batch(
                contexts, horizon=horizon, return_quantiles=True, use_symmetric_averaging=False
            )
        )
        if len(outputs) != len(ids):
            raise ValueError(f"expected {len(ids)} forecasts, got {len(outputs)}")

        frames = []
        for uid, last, out in zip(ids, last_dates, outputs, strict=True):
            dates = pd.date_range(last, periods=horizon + 1, freq=self.freq)[1:]
            frame = pd.DataFrame({"unique_id": uid, "ds": dates})
            values = np.asarray(out.quantiles)[:, columns]  # shape (horizon, len(qs))
            for j, q in enumerate(qs):
                frame[f"q{q}"] = values[:, j]
            frames.append(frame)
        return finalize_quantiles(pd.concat(frames, ignore_index=True), qs)
