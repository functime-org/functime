# Developer Guide

This guide shows you how to use `functime`'s primitives to create new `forecasters` and `transformers`. If you would like to add your custom implementation into the `functime` library, please open up an [draft pull request on GitHub](https://github.com/TracecatHQ/functime/pulls)! All contributions are welcome.

## Build your own `forecaster`

🚧 Under construction.

## Build your own `transformer`

`functime` provides an easy-to-use and functional `@transformer` decorator to implement new `transformers`. Here is an example:

```python
@transformer
def lag(lags: List[int]):
    """Applies lag transformation to a LazyFrame.

    Parameters
    ----------
    lags : List[int]
        A list of lag values to apply.
    """

    def transform(X: pl.LazyFrame) -> pl.LazyFrame:
        entity_col = X.columns[0]
        time_col = X.columns[1]
        max_lag = max(lags)
        lagged_series = [
            (
                pl.all()
                .exclude([entity_col, time_col])
                .shift(lag)
                .over(entity_col)
                .suffix(f"__lag_{lag}")
            )
            for lag in lags
        ]
        X_new = (
            # Pre-sorting seems to improve performance by ~20%
            X.sort(by=[entity_col, time_col])
            .select(
                pl.col(entity_col).set_sorted(),
                pl.col(time_col).set_sorted(),
                *lagged_series,
            )
            .group_by(entity_col)
            .agg(pl.all().slice(max_lag))
            .explode(pl.all().exclude(entity_col))
        )
        artifacts = {"X_new": X_new}
        return artifacts

    return transform
```

Key points to note:

1. Specify all parameters in the outer function.
2. Implement a curried `transform` function inside the outer function that returns a dictionary. This dictionary must contain `X_new` key mapped to the transformed DataFrame. Every `transform` function expects a panel DataFrame.
