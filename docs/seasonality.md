## Seasonality and Holiday Effects

## Modelling Seasonality

### Seasonal Periods

Given a Polars offset alias `freq`, use `functime.offsets.freq_to_sp` to return a list of seasonal periods.

```python
seasonal_periods = {
    "1s": [60, 3_600, 86_400, 604_800, 31_557_600],
    "1m": [60, 1_440, 10_080, 525_960],
    "30m": [48, 336, 17_532],
    "1h": [24, 168, 8_766],
    "1d": [7, 365],
    "1w": [52],
    "1mo": [12],
    "3mo": [4],
    "1y": [1],
}
```

### Method 1. Dummy Variables / Categorical

Use `add_calendar_effects` to generate datetime and calendar effects. `functime` supports two strategies to model seasonality as discrete features: though a categorical column (useful for forecasters with native categorical features support e.g. `lightgbm`) or multiple binary columns (i.e. one-hot encoding). Check out [Chapter 7.4: Seasonal dummy variables](https://otexts.com/fpp3/useful-predictors.html#seasonal-dummy-variables) for a quick primer.

If you choose the dummy variable strategy, beware of the "dummy variable trap" (i.e. remember to set `fit_intercept=False` if you decide to include all dummy columns).

  - minute: 1, 2, ..., 60 (in a day)
  - hour: 1, 2, ..., 24 (in a day)
  - day: 1, 2, ..., 31 (in a month)
  - weekday: 1, 2, ..., 7 (in a week)
  - week: 1, 2,..., 52 (in a year)
  - quarter: 1, 2, ..., 4 (in a year)
  - year: 1999, 2000, ..., 2023 (any year)

```python
from functime.seasonality import add_calendar_effects

# Returns X with one categorical column "month" with values 1,2,...,12
X_new = X.pipe(add_calendar_effects(["month"])).collect()

# Returns X with one-hot encoded calendar effects
# i.e. binary columns "month_1", "month_2", ..., "month_12"
X_new = X.pipe(add_calendar_effects(["month"]), as_dummies=True).collect()
```

### Method 2. Fourier Terms

Fourier terms are a common way to model multiple seasonal periods and complex seasonality (e.g. long seasonal periods 365.25 / 7 ≈ 52.179 for weekly time series). For every seasonal period `sp` and Fourier term `k=1,..,K` pair, there are 2 fourier terms `sin_sp_k` and `cos_sp_k`.

Fourier terms can be used to approximate a continuous periodic signal, which can then be used as exogenous regressors to model seasonality.
[Chapter 12.1: Complex Seasonality](https://otexts.com/fpp3/complexseasonality.html) from Hyndman's textbook "Forecasting: Principles and Practice" contains a great practical introduction to this topic.

`add_fourier_terms` returns the original `X` DataFrame along with the Fourier terms as additional columns.
For example, if `sp=12` and `K=3`, `X_new` would contain the columns `sin_12_1`, `cos_12_1`, `sin_12_2`, `cos_12_2`, `sin_12_3`, and `cos_12_3`.


```python
from functime.offsets import freq_to_sp
from functime.seasonality import add_fourier_terms

sp = freq_to_sp["1mo"][0]
X_new = X.pipe(add_fourier_terms(sp=sp, K=3)).collect()
```

## Modelling Holidays / Special Events

`functime` has a wrapper function around the [`holidays`](https://pypi.org/project/holidays/) Python package to generate categorical features for special events. Dates without a holiday are filled with nulls.

```python
from functime.seasonality import add_holiday_effects

# Returns X with two categorical columns "holiday__US" and "holiday__CA"
north_america_holidays = add_holiday_effects(country_codes=["US", "CA"])
X_new = X.pipe(north_america_holidays).collect()

# Returns X with one-hot encoded holidays (e.g. "holiday__US_christmas)
north_america_holidays = add_holiday_effects(country_codes=["US", "CA"], as_dummies=True)
X_new = X.pipe(north_america_holidays).collect()
```

!!! tip "Custom Events"

    If you have your own custom special events (e.g. special promotions), you can always create your own [dummy variables](https://otexts.com/fpp3/useful-predictors.html#dummy-variables) as Polars [boolean series](https://pola-rs.github.io/polars-book/user-guide/expressions/casting/#booleans).
