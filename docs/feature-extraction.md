# Feature Extraction

`functime` has over 100+ time-series feature extractors (e.g. `binned_entropy`, `longest_streak_above_mean`) available for any `Polars` Series. Approximately 85% of the implementations are optimized lazy queries and works on both `polars.Series` and `polars.Expr`.

Check out the [API reference](ref/feature-extraction.md) for a list of supported feature extractors.

## Usage Examples

Every feature is easily accessible via `functime`'s custom `ts` (time-series) namespace, which works with any `Polars` Series or expression. To register the custom `ts` `Polars` namespace, you must first import `functime`.

```python
import polars as pl
import numpy as np
from functime.feature_extractors import FeatureExtractor, binned_entropy

# Load commodities price data
y = pl.read_parquet("https://github.com/TracecatHQ/functime/raw/main/data/commodities.parquet")

# Get column names ("commodity_type", "time", "price")
entity_col, time_col, value_col = y.columns

# Extract a single feature from a single time-series
binned_entropy = binned_entropy(
    pl.Series(np.random.normal(0, 1, size=10)),
    bin_count=10
)

# 🔥 Also works on LazyFrames with query optimization
features = (
    pl.LazyFrame({
        "index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "value": np.random.normal(0, 1, size=10)
    })
    .select(
        binned_entropy=pl.col("value").ts.binned_entropy(bin_count=10),
        lempel_ziv_complexity=pl.col("value").ts.lempel_ziv_complexity(threshold=3),
        longest_streak_above_mean=pl.col("value").ts.longest_streak_above_mean(),
    )
    .collect()
)

# 🚄 Extract features blazingly fast on many
# stacked time-series using `group_by`
features = (
    y.group_by(entity_col)
    .agg(
        binned_entropy=pl.col(value_col).ts.binned_entropy(bin_count=10),
        lempel_ziv_complexity=pl.col(value_col).ts.lempel_ziv_complexity(threshold=3),
        longest_streak_above_mean=pl.col(value_col).ts.longest_streak_above_mean(),
    )
)

# 🚄 Extract features blazingly fast on windows
# of many time-series using `group_by_dynamic`
features = (
    # Compute rolling features at yearly intervals
    y.group_by_dynamic(
        time_col,
        every="12mo",
        by=entity_col,
    )
    .agg(
        binned_entropy=pl.col(value_col).ts.binned_entropy(bin_count=10),
        lempel_ziv_complexity=pl.col(value_col).ts.lempel_ziv_complexity(threshold=3),
        longest_streak_above_mean=pl.col(value_col).ts.longest_streak_above_mean(),
    )
)
```
