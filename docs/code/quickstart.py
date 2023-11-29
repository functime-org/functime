import json
from timeit import default_timer

import polars as pl

from functime.cross_validation import train_test_split
from functime.forecasting import auto_linear_model, linear_model, naive, snaive
from functime.metrics import smape
from functime.preprocessing import scale
from functime.seasonality import add_fourier_terms

start_time = default_timer()

# Load data
y = pl.read_parquet(
    "https://github.com/TracecatHQ/functime/raw/main/data/commodities.parquet"
)
entity_col, time_col = y.columns[:2]
X = y.select([entity_col, time_col]).pipe(add_fourier_terms(sp=12, K=6)).collect()

print("🎯 Target variable (y):\n", y)
print("📉 Exogenous variables (X):\n", X)

# Train-test splits
test_size = 3
freq = "1mo"
y_train, y_test = train_test_split(test_size)(y)
X_train, X_test = train_test_split(test_size)(X)

# Paralleized naive forecasts!
y_pred_naive = naive(freq="1mo")(y=y_train, fh=3)
y_pred_snaive = snaive(freq="1mo", sp=12)(y=y_train, fh=3)


# Univariate time-series fit with automated lags and hyperparameter tuning
auto_forecaster = auto_linear_model(
    freq=freq, test_size=test_size, min_lags=12, max_lags=18, n_splits=3, time_budget=3
)
auto_forecaster.fit(y=y_train)
# Predict
y_pred = auto_forecaster.predict(fh=test_size)
# Score
scores = smape(y_true=y_test, y_pred=y_pred)
print("✅ Predictions (univariate):\n", y_pred.sort(entity_col))
print("💯 Scores (univariate):\n", scores.sort("smape"))
print("💯 Scores summary (univariate):\n", scores.select("smape").describe())

# Retrieve best lags and hyperparameters
best_params = auto_forecaster.best_params
print(f"✨ Best parameters (y only):\n{json.dumps(best_params, indent=4)}")

# Multivariate
forecaster = linear_model(**best_params)
forecaster.fit(y=y_train, X=X_train)
# Predict
y_pred = forecaster.predict(fh=test_size, X=X_test)
# Score
scores_with_exog = smape(y_true=y_test, y_pred=y_pred)

print("✅ Predictions (multivariate):\n", y_pred.sort(entity_col))
print("💯 Scores (multivariate):\n", scores_with_exog.sort("smape"))
print(
    "💯 Scores summary (multivariate):\n", scores_with_exog.select("smape").describe()
)

# Check uplift from Fourier features
uplift = (
    scores_with_exog.join(scores, on=entity_col, suffix="_univar")
    .with_columns(
        uplift=pl.col("smape_univar") - pl.col("smape"),
        has_uplift=pl.col("smape_univar") - pl.col("smape") > 0,
    )
    .select([entity_col, "uplift", "has_uplift"])
)

# NOTE: Fourier features lead to uplift for ~20% of commodities
# However, at the expense of an overall mean and variance SMAPE
# (likely due to overfitting on seasonal features)

print("💯 Uplift:\n", uplift.sort("uplift", descending=True))
print("💯 Proportion with uplift:", uplift.get_column("has_uplift").mean())

# "Direct" strategy forecasting
best_params["max_horizons"] = test_size  # Override max_horizons
best_params["strategy"] = "direct"  # Override strategy
# Predict using the "functional" API
y_pred = linear_model(**best_params)(y=y_train, fh=test_size)

# "Ensemble" strategy forecasting
best_params["strategy"] = "ensemble"  # Override strategy

# Backtesting
y_preds = linear_model(**best_params).backtest(y=y_train, X=X_train)
print("✅ Backtests:", y_preds)

# Forecast with target transforms and feature transforms
forecaster = linear_model(
    freq="1mo",
    lags=24,
    target_transform=scale(),
    feature_transform=add_fourier_terms(sp=12, K=6),
)
y_pred = forecaster(y=y_train, fh=test_size)

elapsed_time = default_timer() - start_time
print(f"⏱️ Elapsed time: {elapsed_time}")
