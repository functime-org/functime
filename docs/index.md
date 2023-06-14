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

```
### Clustering

```python

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
