# functime

![functime](img/banner.png)

## Production-ready time series models

**functime** is a machine learning library for time-series predictions that [just works](https://www.functime.ai/).

- **Fully-featured:** Powerful and easy-to-use API for [forecasting](#forecasting-highlights) and [feature engineering](#feature-engineering-highlights) (tsfresh, Catch22).
- **Fast:** Forecast [100,000 time series](#global-forecasting) in seconds *on your laptop*
- **Efficient:** Extract 100s of time-series features in parallel using [Polars](https://www.pola.rs/) *
- **Battle-tested:** Algorithms that deliver real business impact and win competitions

## Installation

Check out this [guide](installation.md) to install functime. Requires Python 3.8+.

## Supported Data Schemas

!!! info "Panel Data"

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

!!! info "Time Series"

    Feature extractors support both **panel** and **time-series** DataFrames. Time-series Dataframes represents the measurements for a single entity:

    ```
    >>> y_time_series
    shape: (756, 3)

    time         price
    -------------------
    1960-01-01    511.47
    1960-02-01    511.47
    1960-03-01    511.47
    ...              ...
    2022-11-01   2938.92
    2022-12-01   3129.48
    2023-01-01   3309.81
    ```

## Features

### Forecasting

Point and probablistic forecasts using machine learning.
Includes utilities to support the full forecasting lifecycle: preprocessing, feature extraction, time-series cross-validation / splitters, backtesting, automated hyperparameter tuning, and scoring.

- Every forecaster supports **exogenous features**
- **Seasonality** effects using [calendar, Fourier, and holiday features](https://docs.functime.ai/seasonality/)
- **Backtesting** with [expanding window and sliding window splitters](https://docs.functime.ai/ref/cross-validation/)
- **Automated lags and hyperparameter tuning** using [`FLAML`](https://github.com/microsoft/FLAML)
- **Probablistic forecasts** via quantile regression and conformal prediction
- **Forecast metrics** (e.g. MASE, SMAPE, CRPS) for scoring in parallel
- Supports **recursive and direct** forecast strategies
- **Censored model** for zero-inflated forecasts

View the [full walkthrough](forecasting.md) on forecasting with `functime`.

### Feature Extraction
`functime` has over 100+ time-series feature extractors (e.g. `binned_entropy`, `longest_streak_above_mean`) available for any `Polars` Series. Approximately 85% of the implementations are optimized lazy queries and works on both `polars.Series` and `polars.Expr`.

- Over 100+ time-series features
- All features are registered under a custom `ts` Polars namespace
- ~85% optimized lazy queries and works on both `polars.Series` and `polars.Expr`
- 2x-200x speed-ups compared to `tsfresh`
- >200x speed-ups compared to `tsfresh` for group by operations
- Supports univariate feature extraction
- Supports feature extraction across many time-series (via `group_by`)
- Supports feature extraction across windows (via `group_by_dynamic`)

View the [full walkthrough](feature_extraction.md) on forecasting with `functime`.

### Preprocessing
View API reference for [`functime.preprocessing`](https://docs.functime.ai/preprocessing/).
Preprocessors take in a `polars.DataFrame` or `polars.LazyFrame` as input and **always returns a `polars.LazyFrame`**.
No computation is run until the `.collect()` method is called on the LazyFrame.
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
