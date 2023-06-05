![functime](https://github.com/indexhub-ai/functime/raw/main/static/images/functime_banner.png)

<div align="center">
<h2 align="center">Run and deploy time-series machine learning, remarkably fast</h2>
</div>

`functime` is the world's most powerful and easy-to-use API for AutoML forecasting and time-series embeddings.

Want to use `functime` for seamless time-series analytics across your data team?
Looking for production-grade AI/ML forecasting and time-series search that scales?
Book a [15 minute discovery call](https://calendly.com/functime-indexhub) to learn more about `functime`'s Team / Enterprise plans.

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
from functime.feature_extraction import add_calendar_effects, add_holiday_effects
from functime.forecasting import LightGBM
from functime.metrics import mase

# Specify forecast horizon (the number of periods to predict into the future)
fh = 3
freq = "1mo"

# Load example data
y = pl.read_parquet("https://bit.ly/commodities-data")
entity_col, time_col = y.columns[:2]

# Add calendar and holiday effects
X = (
    y.select([entity_col, time_col])
    .pipe(add_calendar_effects(["month", "year"]))
    .collect()
)

# Time series split
y_train, y_test = y.pipe(train_test_split(test_size=fh))
X_train, X_test = X.pipe(train_test_split(test_size=fh))

# Specify model
model = LightGBM(freq="1mo", lags=12, straight="recursive")

# Fit then predict
model.fit(y=y_train, X=X_train)
y_pred = model.predict(fh=fh, X=X_test)

# Score forecasts in parallel
scores = mase(y_true=y_test, y_pred=y_pred, y_train=y_train)
```
All predictions and scores are returned as `Polars` DataFrames.
```
>>> y_pred

>>> scores
```

## Highlights
- **Fast:** Forecast 100,000 time series in seconds *on your laptop*
- **Efficient:** Embarressingly parallel feature engineering for time-series using [`Polars`](https://www.pola.rs/)
- **Battle-tested:** Automated machine learning algorithms that deliver real business impact and win competitions
- Every forecaster supports **exogenous features**
- **Backtesting** with expanding window and sliding window splitters
- Automated lags and **hyperparameter tuning** using [`FLAML`](https://github.com/microsoft/FLAML)
- Utilities to add calendar effects, special events (e.g. holidays), weather patterns, and economic trends
- Supports recursive, direct, and ensemble forecast strategies

View detailed [list of features](https://docs.functime.ai/features/) including forecasters, preprocessors, feature extractors, and time-series splitters.

## Deployment
`functime` deploys and trains your forecasting models the moment you call any `.fit` method.
Run the `functime list` CLI command to list all deployed models.
```bash
```

You can reuse a deployed model for predictions anywhere using the `stub_id` variable.
```python
```

## License
`functime` is distributed under [AGPL-3.0-only](LICENSE). For Apache-2.0 exceptions, see [LICENSING.md](https://github.com/indexhub-ai/functime/blob/HEAD/LICENSING.md).
