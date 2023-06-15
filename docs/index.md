# functime

![functime](img/banner.png)

## Production-ready time series models

**functime** is a machine learning library for time-series predictions that [just works](https://www.functime.ai/).

- **Fully-featured:** Powerful and easy-to-use API for [AutoML forecasting](#forecasting-highlights) and [time-series embeddings](#embeddings-highlights) (classification, anomaly detection, and clustering)
- **Fast:** Forecast and classify [100,000 time series](#global-forecasting) in seconds *on your laptop*
- **Cloud-native:** Instantly run, deploy, and serve predictive time-series models
- **Efficient:** Embarressingly parallel feature engineering using [Polars](https://www.pola.rs/) *
- **Battle-tested:** Algorithms that deliver real business impact and win competitions

## Installation

Check out this [guide](installation.md) to install functime. Requires Python 3.8+.

## Forecasting

Point and probablistic forecasts using machine learning.
Includes utilities to support the full forecasting lifecycle:
preprocessing, feature extraction, time-series cross-validation / splitters, backtesting, AutoML, hyperparameter tuning, and scoring.

- Every forecaster supports **exogenous features**
- **Backtesting** with expanding window and sliding window splitters
- **Automated lags and hyperparameter tuning** using [`FLAML`](https://github.com/microsoft/FLAML)
- **Probablistic forecasts** via quantile regression and conformal prediction
- **Forecast metrics** (e.g. MASE, SMAPE, CRPS) for scoring in parallel
- Supports **recursive and direct** forecast strategies

View the [full walkthrough](forecasting.md) on forecasting with `functime`.

## Embeddings

Time-series embeddings measure the relatedness of time-series.
Embeddings are more accurate and efficient compared to statistical methods (e.g. Catch22) for characteristing time-series.[^1]
Embeddings have applications across many domains from finance to IoT monitoring.
They are commonly used for the following tasks:

- **Search:** Where time-series are ranked by similarity to a given time-series
- **Classification:** Where time-series are grouped together by matching patterns
- **Clustering:** Where time-series are assigned labels (e.g. normal vs irregular heart rate)
- **Anomaly detection:** Where outliers with unexpected regime / trend changes are identified

View the [full walkthrough](embeddings.md) on time-series embeddings with `functime`.

[^1]: Middlehurst, M., Schäfer, P., & Bagnall, A. (2023). Bake off redux: a review and experimental evaluation of recent time series classification algorithms. arXiv preprint arXiv:2304.13029.

## Quick Examples

### Forecasting

```python
import polars as pl
from functime.cross_validation import train_test_split
from functime.forecasting import LightGBM
from functime.metrics import mase

# Load example data in "panel" format:
# Column 1. Entity (e.g. commodity name)
# Column 2. Time (e.g. date)
# Column 3. Value (e.g. price)
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
import numpy as np
import polars as pl
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from functime.embeddings import embed

# Load GunPoint dataset
data_url = "https://github.com/indexhub-ai/functime/raw/main/data"

# Train-test split
X_y_train = pl.read_parquet(f"{data_url}/gunpoint_train.parquet")
X_y_test = pl.read_parquet(f"{data_url}/gunpoint_test.parquet")

X_train = X_y_train.select(pl.all().exclude("label"))
y_train = X_y_train.select("label")
X_test = X_y_test.select(pl.all().exclude("label"))
y_test = X_y_test.select("label")

# `embed()` takes in a list of time series as a 2D numpy array
# The transformation returns an embedding for each time series
# i.e. [[ts-0], ... [ts-N]] -> [[emb-0], ... [emb-N]]
X_train_embs = embed(X_train, model="minirocket")

# The training embeddings ndarray has the shape:
# (Number of training time series, Closest multiple of 84 < 10,000 => 9996)
# where ~10,000 is the recommended number of features
np.testing.assert_equal(X_train_embs.shape, (len(X_train), 9_996))

# Fit an sklearn classifier on the embeddings
classifier = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
)
classifier.fit(X_train_embs, y_train)

X_test_embs = embed(X_test, model="minirocket")

# Similarly, the test embeddings ndarray has the shape:
# (Number of test time series, Closest multiple of 84 < 10,000 => 9996)
np.testing.assert_equal(X_test_embs.shape, (len(X_test), 9_996))

# Predict (alternatively: 'classifier.score(X_test_embs, y_test)')
predictions = classifier.predict(X_test_embs)
accuracy = accuracy_score(predictions, y_test)
```
### Clustering

```python
import numpy as np
import polars as pl
import pandas as pd
import pyarrow as pa
import yfinance as yf
import requests
import hdbscan
import umap
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from functime.embeddings import embed
from functime.preprocessing import roll


# Download S&P 500 stock prices
start_date = "2022-06-01"
end_date = "2023-06-01"

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
response = requests.get(
    url,
    headers={
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    },
)
data = pd.read_html(response.text)[0]
tickers = [sym.replace(".", "-") for sym in data["Symbol"].to_list()]
stock_prices = yf.download(
    tickers, start_date, end_date, auto_adjust=True
)["Close"]

# Replace the column names with a more readable format
schema = [f"{sec} ({sect})" for sec, sect in zip(securities, sectors)]
df = pl.DataFrame(stock_prices, schema=schema).fill_null(strategy="zero")

# Reduce noise by smoothing the time series. We first `melt` the wide format
# data into long (panel) format to use with `roll()`, as it expects data in
# the format: [entity_col, time_col, target_col]
df_melted = (
    df.hstack([pl.Series("time", stock_prices.index)])
    .melt(id_vars="time", variable_name="ticker", value_name="price")
    .select(["ticker", "time", "price"])
)
smoothed_df = (
    df_melted.pipe(roll(window_sizes=[60], stats=["mean"], freq="1d"))
    .drop_nulls()
    .collect()
)

# Return the data back into wide format
df = (
    smoothed_df.pivot(
        values=f"price__rolling_mean_{window_sz}",
        columns="ticker",
        index="time"
    )
    .sort(by=pl.col("time"))
    .drop("time")
)

# Create embeddings
# As Polars dataframes are built on Apache Arrow's columnar memory format,
# we need to transpose the ndarray into row-major format for `embed()`
X = df.to_numpy()
embeddings = embed(X.T, model="minirocket")


# Reduce dimensionality with UMAP
reducer = umap.UMAP(
    n_components=500, n_neighbors=10, metric="manhattan", random_state=0
)
umap_embeddings = reducer.fit_transform(embeddings)
clusterer = hdbscan.HDBSCAN(gen_min_span_tree=True, metric="minkowski", p=1)

# Cluster with HDBSCAN
# We use GridSearchCV to find the best parameters for the clusterer
def hdbscan_scorer(estimator, X_):
    estimator.fit(X_)
    return estimator.relative_validity_

params = {
    "min_samples": np.arange(5, 30),
    "min_cluster_size": np.arange(5, 30),
}
grid = GridSearchCV(
    clusterer, param_grid=params, scoring=hdbscan_scorer, n_jobs=-1, cv=5
)
grid.fit(umap_embeddings)

# View best clusterer parameters
estimator = grid.best_estimator_
labels = estimator.labels_
cluster_labels = np.unique(labels)
n_labels = len(cluster_labels)
n_clustered = np.sum((labels >= 0))
coverage = n_clustered / umap_embeddings.shape[0]
```

### Preprocessing

## Time Series Data

### External Data

Easily enrich your own data with `functime`'s built-in datasets. Datasets include calendar effects, holidays, weather patterns, economic data, and seasonality features (i.e. Fourier Series).

### Example Data

It is easy to get started with `functime`.
Our GitHub repo contains a growing number of time-series data stored as `parquet` files:

- M4 Competition (daily, weekly, monthly, quarterly, yearly)[^2]
- M5 Competition[^3]
- Australian tourism[^4]
- Commodities prices[^5]
- Gunpoint measurements[^6]
- Japanese vowels [^7]

[^2]: https://mofc.unic.ac.cy/m4/
[^3]: https://mofc.unic.ac.cy/m5-competition/
[^4]: https://www.abs.gov.au/statistics/industry/tourism-and-transport/overseas-arrivals-and-departures-australia
[^5]: https://www.imf.org/en/Research/commodity-prices
[^6]: http://www.timeseriesclassification.com/description.php?Dataset=GunPoint
[^7]: http://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels

## Contact Us

Book a quick 15 minute discovery call on [Calendly](https://calendly.com/functime-indexhub).
