<div align="center">
    <h1>Run and scale time-series machine learning</h1>
<br />

![functime](https://github.com/indexhub-ai/functime/raw/main/static/images/functime_banner.png)
[![Python](https://img.shields.io/pypi/pyversions/functime)](https://pypi.org/project/functime/)
[![PyPi](https://img.shields.io/pypi/v/functime?color=blue)](https://pypi.org/project/functime/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub Publish to PyPI](https://github.com/indexhub-ai/functime/actions/workflows/publish.yml/badge.svg)](https://github.com/indexhub-ai/functime/actions/workflows/publish.yml)
[![GitHub Build Docs](https://github.com/indexhub-ai/functime/actions/workflows/docs.yml/badge.svg)](https://docs.functime.ai/)
[![GitHub Run Quickstart](https://github.com/indexhub-ai/functime/actions/workflows/quickstart.yml/badge.svg)](https://github.com/indexhub-ai/functime/actions/workflows/quickstart.yml)

</div>

---
**functime** is a powerful and easy-to-use [Cloud service](https://functime.ai) for AutoML forecasting and time-series embeddings.
The `functime` [Python package](https://pypi.org/project/functime/) provides a scikit-learn API and command-line interface to interact with **functime Cloud**.

**functime** also comes with open-sourced [Apache 2.0](https://github.com/indexhub-ai/functime/blob/HEAD/LICENSING.md) time-series [preprocessing](https://docs.functime.ai/ref/preprocessing/) (box-cox, differencing etc), cross-validation [splitters](https://docs.functime.ai/ref/cross-validation/) (expanding and sliding window), and forecast [metrics](https://docs.functime.ai/ref/metrics/) (MASE, SMAPE etc). All optimized as [lazy Polars](https://pola-rs.github.io/polars-book/user-guide/lazy/using/) transforms.

Want to use **functime** for seamless time-series analytics across your data team
Looking for fully-managed production-grade AI/ML forecasting and time-series embeddings?
Book a [15 minute discovery call](https://calendly.com/functime-indexhub) to learn more about functime's Team / Enterprise plans.

## Highlights
- **Fast:** Forecast 100,000 time series in seconds *on your laptop*
- **Efficient:** Embarrassingly parallel [feature engineering](https://docs.functime.ai/ref/preprocessing/) for time-series using [`Polars`](https://www.pola.rs/)
- **Battle-tested:** Machine learning algorithms that deliver real business impact and win competitions
- **Exogenous features:** supported by every forecaster
- **Backtesting** with expanding window and sliding window splitters
- **AutoML**: Automated lags and hyperparameter tuning using [`FLAML`](https://github.com/microsoft/FLAML)
- Utilities to add calendar effects, special events (e.g. holidays), weather patterns, and economic trends
- Supports recursive, direct, and ensemble forecast strategies

**Note:** All preprocessors, time-series splitters, and forecasting metrics are implemented with [`Polars`](https://www.pola.rs/) and open-sourced under the Apache-2.0 license. Contributions are always welcome.

## Getting Started
1. First, install `functime` via the [pip](https://pypi.org/project/functime) package manager.
```bash
pip install functime
```
2. Then sign-up for a free `functime` Cloud account via the command-line interface (CLI).
```bash
functime login
```
3. That's it! You can execute time series predictions at scale using functime's `scikit-learn` fit-predict API.

### Forecasting

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

# functime ❤️ functional design
# fit-predict in a single line
y_pred = LightGBM(freq="1mo", lags=24)(y=y_train, fh=3)

# Score forecasts in parallel
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
```

### Classification

```python
import polars as pl
import functime
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# Load GunPoint dataset (150 observations, 150 timestamps)
X_y_train = pl.read_parquet("https://bit.ly/gunpoint-train")
X_y_test = pl.read_parquet("https://bit.ly/gunpoint-test")

# Train-test split
X_train, y_train = X_y_train.select(pl.all().exclude("label")), X_y_train.select("label")
X_test, y_test = X_y_test.select(pl.all().exclude("label")), X_y_test.select("label")

X_train_embs = functime.embeddings.embed(X_train, model="minirocket")

# Fit classifier on the embeddings
classifier = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
)
classifier.fit(X_train_embs, y_train)

# Predict and
X_test_embs = embed(X_test, model="minirocket")
labels = classifier.predict(X_test_embs)
accuracy = accuracy_score(predictions, y_test)
```

### Clustering

```python
import functime
import polars as pl
from hdbscan import HDBSCAN
from umap import UMAP
from functime.preprocessing import roll

# Load S&P500 panel data from 2022-06-01 to 2023-06-01
# Columns: ticker, time, price
y = pl.read_parquet("https://bit.ly/sp500-data")

# Create embeddings
embeddings = functime.embeddings.embed(y_ma_60, model="minirocket")

# Reduce dimensionality with UMAP
reducer = UMAP(n_components=500, n_neighbors=10, metric="manhattan")
umap_embeddings = reducer.fit_transform(embeddings)

# Cluster with HDBSCAN
clusterer = HDBSCAN(metric="minkowski", p=1)
estimator.fit(X)

# Get predicted cluster labels
labels = estimator.predict(X)
```

## Deployment
`functime` deploys and trains your forecasting models the moment you call any `.fit` method.
Run the `functime list` CLI command to list all deployed models.
To view data and forecasts usage, run the `functime usage` CLI command.

![Example CLI usage](static/gifs/functime_cli_usage.gif)

You can reuse a deployed model for predictions anywhere using the `stub_id` variable.
Note: the `.from_deployed` model class must be the same as during `.fit`.
```python
forecaster = LightGBM.from_deployed(stub_id)
y_pred = forecaster.predict(fh=3)
```

## License
`functime` is distributed under [AGPL-3.0-only](LICENSE). For Apache-2.0 exceptions, see [LICENSING.md](https://github.com/indexhub-ai/functime/blob/HEAD/LICENSING.md).
