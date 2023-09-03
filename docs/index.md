# functime

![functime](img/banner.png)

## Production-ready time series models

**functime** is a machine learning library for time-series predictions that [just works](https://www.functime.ai/).

- **Fully-featured:** Powerful and easy-to-use API for [forecasting](#forecasting-highlights) and [feature engineering](#feature-engineering-highlights) (tsfresh, Catch22).
- **Fast:** Forecast and classify [100,000 time series](#global-forecasting) in seconds *on your laptop*
- **Cloud-native:** Instantly run, deploy, and serve predictive time-series models
- **Efficient:** Embarressingly parallel feature engineering using [Polars](https://www.pola.rs/) *
- **Battle-tested:** Algorithms that deliver real business impact and win competitions

## Installation

Check out this [guide](installation.md) to install functime. Requires Python 3.8+.

## Forecasting

Point and probablistic forecasts using machine learning.
Includes utilities to support the full forecasting lifecycle:
preprocessing, feature extraction, time-series cross-validation / splitters, backtesting, automated hyperparameter tuning, and scoring.

- Every forecaster supports **exogenous features**
- **Backtesting** with expanding window and sliding window splitters
- **Automated lags and hyperparameter tuning** using [`FLAML`](https://github.com/microsoft/FLAML)
- **Probablistic forecasts** via quantile regression and conformal prediction
- **Forecast metrics** (e.g. MASE, SMAPE, CRPS) for scoring in parallel
- Supports **recursive and direct** forecast strategies
- **Censored model** for zero-inflated forecasts

View the [full walkthrough](forecasting.md) on forecasting with `functime`.

## Quick Examples

!!! info "Input Data Schemas"

    Forecasters, preprocessors, and splitters take a **panel dataset** where the first two columns represent entity (e.g. commodty name) and time (e.g. date). Subsequent columns represent observed values (e.g. price). The panel DataFrame **must be sorted by entity, time.**

    ```
    >>> y_panel
    shape: (47_583, 3)

    commodity_type   time         price
    ------------------------------------
    Aluminum         1960-01-01    511.47
                     1960-02-01    511.47
                     1960-03-01    511.47
                     1960-04-01    511.47
                     1960-05-01    511.47
    ...                     ...       ...
    Zinc             2022-11-01   2938.92
                     2022-12-01   3129.48
                     2023-01-01   3309.81
                     2023-02-01   3133.84
                     2023-03-01   2967.46
    ```

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
```

### Splitters
View API reference for [`functime.cross_validation`](https://docs.functime.ai/ref/cross-validation/).
`functime` currently supports expanding window and rolling window splitters.
Splitters are used for cross-validation and backtesting.

### Preprocessing
View API reference for [`functime.preprocessing`](https://docs.functime.ai/ref/cross-validation/).
Preprocessors take in a `polars.DataFrame` or `polars.LazyFrame` as input and **always returns a `polars.LazyFrame`**.
No computation is run until the collect() method is called on the LazyFrame.
This allows Polars to [optimize the whole query](https://pola-rs.github.io/polars-book/user-guide/lazy/optimizations/) before execution.

```python
from functime.preprocessing import boxcox, impute

# Use df.pipe to chain operations together
X_new: pl.LazyFrame = (
    X.pipe(boxcox(method="mle"))
    .pipe(detrend(method="linear"))
)
# Call .collect to execute query
X_new: pl.DataFrame = X_new.collect(streaming=True)
```

View [quick examples](preprocessing.md) of time-series preprocessing with `functime`.

## Time Series Data

### Feature Engineering

Easily enrich your forecasts with calendar effects, holidays, weather patterns (coming soon), economic data (coming soon), and seasonality features (i.e. Fourier Series).
View API reference for [`functime.feature_extraction`](https://docs.functime.ai/ref/feature-extraction/).

### Example Data

It is easy to get started with `functime`.
Our GitHub repo contains a growing number of time-series data stored as `parquet` files:

- M4 Competition (daily, weekly, monthly, quarterly, yearly)[^2]
- M5 Competition[^3]
- Australian tourism[^4]
- Commodities prices[^5]
- User laptop activity[^6]
- Gunpoint measurements[^7]
- Japanese vowels [^8]

[^2]: https://mofc.unic.ac.cy/m4/
[^3]: https://mofc.unic.ac.cy/m5-competition/
[^4]: https://www.abs.gov.au/statistics/industry/tourism-and-transport/overseas-arrivals-and-departures-australia
[^5]: https://www.imf.org/en/Research/commodity-prices
[^6]: https://www.sciencedirect.com/science/article/pii/S2352340920306612
[^7]: http://www.timeseriesclassification.com/description.php?Dataset=GunPoint
[^8]: http://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels
