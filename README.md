<div align="center">
    <h1>Time-series machine learning at scale</h1>
<br />

![functime](https://github.com/descendant-ai/functime/raw/main/docs/img/banner_dark_bg.png)
[![Python](https://img.shields.io/pypi/pyversions/functime)](https://pypi.org/project/functime/)
[![PyPi](https://img.shields.io/pypi/v/functime?color=blue)](https://pypi.org/project/functime/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub Publish to PyPI](https://github.com/descendant-ai/functime/actions/workflows/publish.yml/badge.svg)](https://github.com/descendant-ai/functime/actions/workflows/publish.yml)
[![GitHub Run Quickstart](https://github.com/descendant-ai/functime/actions/workflows/quickstart.yml/badge.svg)](https://github.com/descendant-ai/functime/actions/workflows/quickstart.yml)
[![Discord](https://img.shields.io/discord/1145819725276917782)](https://discord.gg/PFJpFqbu4)

</div>

---
**functime** is a powerful [Python library](https://pypi.org/project/functime/) for production-ready **global forecasting** and **time-series feature extraction** on **large panel datasets**.

**functime** also comes with time-series [preprocessing](https://docs.functime.ai/ref/preprocessing/) (box-cox, differencing etc), cross-validation [splitters](https://docs.functime.ai/ref/cross-validation/) (expanding and sliding window), and forecast [metrics](https://docs.functime.ai/ref/metrics/) (MASE, SMAPE etc). All optimized as [lazy Polars](https://pola-rs.github.io/polars-book/user-guide/lazy/using/) transforms.

Join us on [Discord](https://discord.gg/PFJpFqbu4)!

## Highlights
- **Fast:** Forecast and extract features (e.g. tsfresh, Catch22) across 100,000 time series in seconds *on your laptop*
- **Efficient:** Embarrassingly parallel [feature engineering](https://docs.functime.ai/ref/preprocessing/) for time-series using [`Polars`](https://www.pola.rs/)
- **Battle-tested:** Machine learning algorithms that deliver real business impact and win competitions
- **Exogenous features:** supported by every forecaster
- **Backtesting** with expanding window and sliding window splitters
- **Automated lags and hyperparameter tuning** using [`FLAML`](https://github.com/microsoft/FLAML)

## Additional Highlights
`functime` comes with a specialized LLM agent to analyze, describe, and compare your forecasts. Check out the walkthrough [here](https://docs.functime.ai/notebooks/llm/).

## Getting Started
Install `functime` via the [pip](https://pypi.org/project/functime) package manager.
```bash
pip install functime
```

`functime` comes with extra options. For example, to install `functime` with large-language model (LLM) and lightgbm features:

```bash
pip install "functime[llm,lgb]"
```

- `cat`: To use `catboost` forecaster
- `xgb`: To use `xgboost` forecaster
- `lgb`: To use `lightgbm` forecaster
- `llm`: To use the LLM-powered forecast analyst

### Forecasting

```python
import polars as pl
from functime.cross_validation import train_test_split
from functime.feature_extraction import add_fourier_terms
from functime.forecasting import linear_model
from functime.preprocessing import scale
from functime.metrics import mase

# Load commodities price data
y = pl.read_parquet("https://github.com/descendant-ai/functime/raw/main/data/commodities.parquet")
entity_col, time_col = y.columns[:2]

# Time series split
y_train, y_test = y.pipe(train_test_split(test_size=3))

# Fit-predict
forecaster = linear_model(freq="1mo", lags=24)
forecaster.fit(y=y_train)
y_pred = forecaster.predict(fh=3)

# functime ❤️ functional design
# fit-predict in a single line
y_pred = linear_model(freq="1mo", lags=24)(y=y_train, fh=3)

# Score forecasts in parallel
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)

# Forecast with target transforms and feature transforms
forecaster = linear_model(
    freq="1mo",
    lags=24,
    target_transform=scale(),
    feature_transform=add_fourier_terms(sp=12, K=6)
)

# Forecast with exogenous regressors!
# Just pass them into X
X = (
    y.select([entity_col, time_col])
    .pipe(add_fourier_terms(sp=12, K=6)).collect()
)
X_train, X_future = y.pipe(train_test_split(test_size=3))
forecaster = linear_model(freq="1mo", lags=24)
forecaster.fit(y=y_train, X=X_train)
y_pred = forecaster.predict(fh=3, X=X_future)
```

View the full walkthrough on forecasting [here](https://docs.functime.ai/forecasting/).

## LLM Copilot

Requires an OpenAI API key set as an environment variable `OPENAI_API_KEY`.

```python
import polars as pl
import functime.llm

y = pl.read_parquet("../../data/commodities.parquet")
context = "This dataset comprises of historical commodity prices between 1980 to 2022."

# Analyze trend and seasonality for two commodities
analysis = y_pred.llm.analyze(
    context=dataset_context,
    basket=["Aluminum", "Banana, Europe"]
)
print("📊 Analysis:\n", analysis)

# Compare two baskets of commodities!
basket_a = ["Aluminum", "Banana, Europe"]
basket_b = ["Chicken", "Cocoa"]
comparison = y_pred.llm.compare(
    basket=basket_a,
    other_basket=basket_b
)
print("📊 Comparison:\n", comparison)
```

View the full walkthrough on the LLM copilot [here](https://docs.functime.ai/notebooks/llm/).

## License
`functime` is distributed under [Apache-2.0](LICENSE).
