# Preprocessing

`functime` supports parallelized time-series preprocessing using Polars. All `functime` preprocessors take a [panel DataFrame](/#quick-examples) as a input and transform each time-series locally (i.e. time-series by time-series as a parallelized group_by operation).

Time-series transformations are commonly used to stabilize the time-series (e.g. `boxcox` for variance stabilzation) or make the time-series [stationary](https://otexts.com/fpp3/stationarity.html) through first differences or detrending. Some transformations are also invertible, such as `diff` and `detrend`, which is useful for converting the forecast of a transformed time-series back to the original scale.

Check out the [API reference](ref/preprocessing.md) for details.

## Quick Examples

### Differencing

Apply k-order differences. This transform is invertible.

```python
from functime.preprocessing import diff

transformer = diff(order=1)
X_new = X.pipe(transformer).collect()
X_original = transformer.invert(X_new)
```

### Seasonal Differencing

Apply k-order differences shifted by `sp` periods. This transform is invertible.

```python
from functime.preprocessing import diff

# Assume X is a monthly dataset with seasonal period = 12
transformer = diff(order=1, sp=12)
X_new = X.pipe(transformer).collect()
X_original = transformer.invert(X_new)
```

### Detrending (Linear)

Removes linear trend for each time-series. This transform is invertible.

```python
from functime.preprocessing import detrend

transformer = detrend(method="linear")
X_new = X.pipe(transformer).collect()
X_original = transformer.invert(X_new)
```

### Detrending (Mean)

Removes mean trend for each time-series. This transform is invertible.

```python
from functime.preprocessing import detrend

transformer = detrend(method="mean")
X_new = X.pipe(transformer).collect()
X_original = transformer.invert(X_new)
```


### Box-Cox

Applies optimized Box-Cox transform for each time-series. This transform is invertible.

```python
from functime.preprocessing import boxcox

transformer = boxcox(method="mle")
X_new = X.pipe(transformer).collect()
X_original = transformer.invert(X_new)
```


### Yeo-Johnson

Applies optimized Yeo-Johnson transform for each time-series. This transform is invertible.

```python
from functime.preprocessing import yeojohnson
transformer = yeojohnson()
X_new = X.pipe(transformer).collect()
X_original = transformer.invert(X_new)
```


### Local Scaling

Standardizes each time-series with subtracting mean and dividing by the standard deviation. This transform is invertible.

```python
from functime.preprocessing import scale

transformer = scale(use_mean=True, use_std=True)
X_new = X.pipe(transformer).collect()
X_original = transformer.invert(X_new)
```

### Rolling Statistics

Given a list of window sizes, applies rolling statistics for each time-series across each column. This transform is not invertible. Currently supports the following statistics: `mean`, `min`, `max`, `mlm` (max less min), `sum`, `std`, `cv` (coefficient of variation).

```python
from functime.preprocessing import roll

# The following code generates moving averages (MA10, MA30 and MA60)
# and moving sums for a panel dataset of daily time-series.

transformer = roll(
    window_sizes=[10, 30, 60],
    stats=["mean", "sum"],
    freq="1d"
)
X_new = X.pipe(transformer).collect()
```
