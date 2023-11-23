import numpy as np
import polars as pl


def test__cusum_filter():
    vals = list(np.random.default_rng().normal(0.0, 0.1, 150))
    vals_2 = list(np.random.default_rng().normal(0.2, 0.2, 50))

    df = pl.DataFrame({"data": vals + vals_2})

    df = df.with_columns(
        pl.col("data")
        .ts.cusum_filter(threshold=8.0, drift=1.0, warmup_period=50)
        .alias("cusum_event")
    )
    changepoint = (
        df.with_row_count().filter(pl.col("cusum_event") == 1).select("row_nr").item()
    )
    assert 150 < changepoint < 160
