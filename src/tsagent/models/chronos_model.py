"""Zero-shot Chronos-2 (Amazon) behind the common Forecaster interface."""

from __future__ import annotations

import pandas as pd

from tsagent.models.base import finalize_quantiles


class Chronos2Forecaster:
    name = "chronos2"

    def __init__(self, model_id: str = "amazon/chronos-2", device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device
        self._pipeline = None  # loaded on first use, so importing this module needs no torch

    def _load(self):
        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_id, device_map=self.device
            )
        return self._pipeline

    def forecast(
        self, history: pd.DataFrame, horizon: int, quantiles: list[float]
    ) -> pd.DataFrame:
        qs = sorted(quantiles)
        # Chronos-2 expects its own column names; map ours onto them and back.
        context = history[["unique_id", "ds", "y"]].rename(
            columns={"unique_id": "id", "ds": "timestamp", "y": "target"}
        )
        context["target"] = context["target"].astype(float)

        pred = self._load().predict_df(
            context,
            prediction_length=horizon,
            quantile_levels=qs,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        out = pd.DataFrame(
            {"unique_id": pred["id"].to_numpy(), "ds": pred["timestamp"].to_numpy()}
        )
        for q in qs:
            out[f"q{q}"] = pred[str(q)].to_numpy()  # Chronos names quantile columns "0.1", ...
        return finalize_quantiles(out, qs)
