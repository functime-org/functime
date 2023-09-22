`functime` supports both individual forecasters and forecasters with automated lags / hyperparameter tuning.
Auto-forecasters uses `FLAML` to optimize both hyperparameters and number of lagged dependent variables.
`FLAML` is a [SOTA library](https://github.com/microsoft/FLAML) for automated hyperparameter tuning using the CFO (Frugal Optimization for Cost-related Hyperparamters[^1]) algorithm. All individual forecasters (e.g. `lasso` / `xgboost`) and automated forecasters (e.g. `auto_lasso` and `auto_xgboost`) implement the following API.

## `forecaster`

::: functime.base.forecaster.Forecaster

## `auto_forecaster`

::: functime.forecasting.automl.AutoForecaster

[^1]: https://arxiv.org/abs/2005.01571

::: functime.forecasting
