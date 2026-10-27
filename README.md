# Time-series forecasting benchmark

Honest comparison of modern time-series forecasting models, with an LLM agent on top (work in progress).

The goal is a reproducible project that shows (1) how to compare forecasting models fairly, (2) how to use time-series foundation models in zero-shot mode, and (3) how to use an LLM agent responsibly: the LLM chooses tools and explains results, while all numbers are computed by deterministic code.

## Status

Work in progress.

- [x] M5 loader, rolling-origin backtest without look-ahead, metrics (MASE, scaled quantile loss, interval coverage) with unit tests
- [x] Models behind one interface: seasonal naive, LightGBM (direct multi-horizon quantiles), Chronos-2, TimesFM 3.0
- [x] Bootstrap confidence intervals, per-segment analysis, model-selection analysis
- [x] Docker image for the foundation models (CPU)
- [ ] Second dataset (German hourly electricity load, SMARD)
- [ ] LangGraph agent
- [ ] Streamlit demo
- [ ] German README

## Preliminary results

M5 (Walmart), random sample of 500 series, 8 rolling windows of 28 days. Lower MASE is better; the nominal coverage of the 10–90% interval is 0.80.

| Model | MASE | Scaled quantile loss | Interval coverage |
|---|---|---|---|
| TimesFM 3.0 (zero-shot) | 0.760 | 0.260 | 0.831 |
| Chronos-2 (zero-shot) | 0.763 | 0.262 | 0.786 |
| LightGBM | 0.766 | 0.262 | 0.891 |
| Seasonal naive | 1.062 | 0.368 | 0.870 |

- All three modern models beat the seasonal-naive baseline by a wide margin. They are practically tied with each other: the differences are far smaller than the variation between windows.
- Choosing a model per series from past backtest windows did **not** beat always using TimesFM (MASE +0.009 to +0.022, depending on how many past windows are used; the 95% bootstrap intervals exclude zero).
- Averaged over windows, TimesFM has a slightly lower mean MASE than LightGBM (difference 0.006, 95% bootstrap CI 0.001 to 0.011), yet LightGBM has the lower MASE on 63% of the series: it loses on average because of a minority of series where its error is larger.

Caveats: one dataset, one random sample, and the foundation models may have seen M5 during pretraining, which cannot be verified.

## Method

- Rolling-origin backtest: models only see data up to the cutoff (checked by tests).
- Metrics: MASE (scaled by the in-sample seasonal-naive error), quantile loss scaled the same way, and coverage of the 10–90% interval. Series with a zero scale are excluded and counted.
- Paired comparison with a bootstrap over series, plus a per-segment breakdown (length, intermittency, volume).

## Reproduce

```bash
uv sync
uv run pytest
```

The M5 data is not included. Accept the competition rules on Kaggle, then:

```bash
uv run kaggle competitions download -c m5-forecasting-accuracy -p data/raw
unzip data/raw/m5-forecasting-accuracy.zip -d data/raw/m5
```

Foundation models run inside Docker, because PyTorch has no wheels for Intel Macs:

```bash
docker build -t hail-mary-fm .
docker run --rm -v hail-mary-hf:/models \
  -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
  -v "$PWD/src:/app/src" -v "$PWD/scripts:/app/scripts" \
  hail-mary-fm python scripts/run_compare.py --n-series 500 --n-windows 8 --out results/m5_n500_w8
```

TimesFM 3.0 weights are released under a non-commercial license; check the model card before reusing them.

## Stack

Python, pandas, LightGBM, PyTorch (CPU), Chronos-2, TimesFM 3.0, Docker, pytest, ruff, uv. Planned: LangGraph, Streamlit.
