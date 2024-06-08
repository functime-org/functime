from __future__ import annotations

import numpy as np
import polars as pl


def test__cusum_filter():
    vals = list(np.random.default_rng(seed=0).normal(0.0, 0.1, 150))
    vals_2 = list(np.random.default_rng(seed=0).normal(0.2, 0.2, 50))

    df = pl.DataFrame({"data": vals + vals_2})

    df = df.with_columns(
        pl.col("data")
        .ts.cusum(threshold=5.0, drift=1.0, warmup_period=50)
        .alias("cusum_event")
    )
    changepoint = (
        df.with_row_index().filter(pl.col("cusum_event") == 1).select("row_nr").item()
    )
    assert 150 < changepoint < 160
