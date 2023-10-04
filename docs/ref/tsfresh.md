# tsfresh

`functime` has rewritten most of the time-series features extractors from `tsfresh` into `Polars`.
Approximately 80% of the implementations are optimized lazy queries.

The rest are eager implementations. The overall performance improvements compared to `tsfresh` ranges between 5x to 50x.
Speed ups depend on the size of the input, the feature, and whether common subplan elimination is invoked
(i.e. multiple lazy features are collected together). Moreover, windowed / grouped features in `functime` can be a further 100x faster than `tsfresh`.

## Usage Example

```python
import numpy as np
import polars as pl

from functime.feature_extraction.tsfresh import (
    approximate_entropy
    benford_correlation,
    binned_entropy,
    c3
)

sin_x = np.sin(np.arange(120))

# Pass series directly
entropy = approximate_entropy(
    x=pl.Series("ts", sin_x),
    run_length=5,
    filtering_level=0.0
)

# Lazy operations
features = (
    pl.LazyFrame({"ts": sin_x})
    .select(
        approximate_entropy=approximate_entropy(
            pl.col("ts"),
            run_length=5,
            filtering_level=0.0
        ),
        benford_correlation=benford_correlation(pl.col("ts")),
        binned_entropy=binned_entropy(pl.col("ts"), bin_count=10),
        c3=c3(),
    )
    .collect()
)
```

---

::: functime.feature_extraction.tsfresh
