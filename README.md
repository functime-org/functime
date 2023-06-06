<div align="center">
    <h1>Run and deploy time-series machine learning, remarkably fast</h1>
<br />

![functime](https://github.com/indexhub-ai/functime/raw/main/static/images/functime_banner.png)

[![Python](https://img.shields.io/pypi/pyversions/functime-client)](https://pypi.org/project/functime-client/)
[![PyPi](https://img.shields.io/pypi/v/functime-client?color=blue)](https://pypi.org/project/functime-client/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub Publish to PyPI](https://github.com/indexhub-ai/functime/actions/workflows/publish.yml/badge.svg)](https://github.com/indexhub-ai/functime/actions/workflows/publish.yml)
[![GitHub Build Docs](https://github.com/indexhub-ai/functime/actions/workflows/docs.yml/badge.svg)](https://github.com/indexhub-ai/functime/actions/workflows/docs.yml)

</div>

---
`functime` is a powerful and easy-to-use API for AutoML forecasting and time-series embeddings.

Want to use `functime` for seamless time-series analytics across your data team?
Looking for production-grade AI/ML forecasting and time-series search that scales?
Book a [15 minute discovery call](https://calendly.com/functime-indexhub) to learn more about `functime`'s Team / Enterprise plans.

## Highlights
- **Fast:** Forecast 100,000 time series in seconds *on your laptop*
- **Efficient:** Embarrassingly parallel feature engineering for time-series using [`Polars`](https://www.pola.rs/)
- **Battle-tested:** Automated machine learning algorithms that deliver real business impact and win competitions
- Every forecaster supports **exogenous features**
- **Backtesting** with expanding window and sliding window splitters
- Automated lags and **hyperparameter tuning** using [`FLAML`](https://github.com/microsoft/FLAML)
- Utilities to add calendar effects, special events (e.g. holidays), weather patterns, and economic trends
- Supports recursive, direct, and ensemble forecast strategies

View detailed [list of features](https://docs.functime.ai/features/) including forecasters, preprocessors, feature extractors, and time-series splitters.

## Getting Started
1. First, install `functime` via the [pip](https://pypi.org/project/functime-client) package manager.
```bash
pip install "functime-client"
```
2. Then sign-up for a free `functime` Cloud account via the command-line interface (CLI).
```bash
functime login
```
3. That's it! You can begin forecasting at scale using the `scikit-learn` fit-predict interface.
```python
import polars as pl
from functime.cross_validation import train_test_split
from functime.forecasting import LightGBM
from functime.metrics import mase

# Load example data
y = pl.read_parquet("https://bit.ly/commodities-data")
entity_col, time_col = y.columns[:2]

# Time series split
y_train, y_test = y.pipe(train_test_split(test_size=3))

# Fit-predict
model = LightGBM(freq="1mo", lags=24, max_horizons=3, strategy="ensemble")
model.fit(y=y_train)
y_pred = model.predict(fh=3)

# Score forecasts in parallel
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
```
All predictions and scores are returned as `Polars` DataFrames.
```
>>> y_pred
shape: (213, 3)
┌────────────────┬─────────────────────┬─────────────┐
│ commodity_type ┆ time                ┆ price       │
│ ---            ┆ ---                 ┆ ---         │
│ str            ┆ datetime[ns]        ┆ f64         │
╞════════════════╪═════════════════════╪═════════════╡
│ Wheat, US HRW  ┆ 2023-01-01 00:00:00 ┆ 240.337497  │
│ Wheat, US HRW  ┆ 2023-02-01 00:00:00 ┆ 250.851552  │
│ Wheat, US HRW  ┆ 2023-03-01 00:00:00 ┆ 252.102028  │
│ Beef           ┆ 2023-01-01 00:00:00 ┆ 4.271976    │
│ …              ┆ …                   ┆ …           │
│ Coconut oil    ┆ 2023-03-01 00:00:00 ┆ 1140.930346 │
│ Copper         ┆ 2023-01-01 00:00:00 ┆ 7329.806663 │
│ Copper         ┆ 2023-02-01 00:00:00 ┆ 7484.565165 │
│ Copper         ┆ 2023-03-01 00:00:00 ┆ 7486.160195 │
└────────────────┴─────────────────────┴─────────────┘

>>> scores.sort("mase")
shape: (71, 2)
┌──────────────────────┬────────────┐
│ commodity_type       ┆ mase       │
│ ---                  ┆ ---        │
│ str                  ┆ f64        │
╞══════════════════════╪════════════╡
│ Rice, Viet Namese 5% ┆ 0.308148   │
│ Palm kernel oil      ┆ 0.554886   │
│ Coconut oil          ┆ 1.051424   │
│ Cocoa                ┆ 1.32211    │
│ …                    ┆ …          │
│ Sugar, US            ┆ 73.346233  │
│ Sugar, world         ┆ 81.304941  │
│ Phosphate rock       ┆ 85.936644  │
│ Sugar, EU            ┆ 170.319435 │
└──────────────────────┴────────────┘
```

## Deployment
`functime` deploys and trains your forecasting models the moment you call any `.fit` method.
Run the `functime list` CLI command to list all deployed models.
```bash
❯ functime list
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Stub ID                              ┃ Model ID     ┃ Model Params ┃ Stats             ┃ Created At ┃ Last Used  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 51a96242-8826-4096-92aa-737f414a8047 │ linear_model │ lags = 12    │ Dataframe: y      │ 2023-06-06 │ 2023-06-06 │
│                                      │              │              │ N bytes = 7342440 │ 15:49:42   │ 15:49:52   │
│                                      │              │              │ N entities = 359  │            │            │
│                                      │              │              │                   │            │            │
│ a246653d-6d7d-45c6-93d2-5dd59b18c16b │ lightgbm     │ lags = 12    │ Dataframe: y      │ 2023-06-06 │ 2023-06-06 │
│                                      │              │              │ N bytes = 7342440 │ 15:57:05   │ 15:59:16   │
│                                      │              │              │ N entities = 359  │            │            │
│                                      │              │              │                   │            │            │
└──────────────────────────────────────┴──────────────┴──────────────┴───────────────────┴────────────┴────────────┘
```

You can reuse a deployed model for predictions anywhere using the `stub_id` variable.
```python
```

```bash
❯ functime usage
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Metric                       ┃ Limit   ┃ Used            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ Data Used (MB)               │ 2500    │ 143.43 (5.74%)  │
│ Forecasts Used (predictions) │ 1000000 │ 258480 (25.85%) │
│ Max Request Size (MB)        │ 250     │ -               │
└──────────────────────────────┴─────────┴─────────────────┘
```

## License
`functime` is distributed under [AGPL-3.0-only](LICENSE). For Apache-2.0 exceptions, see [LICENSING.md](https://github.com/indexhub-ai/functime/blob/HEAD/LICENSING.md).
