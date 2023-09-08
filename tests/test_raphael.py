import polars as pl
from polars.testing import assert_frame_equal

from functime.feature_extraction.features_raphael import (
    change_quantiles,
    mean_abs_change,
)


def test_change_quantiles():
    df = pl.DataFrame({"value": list(range(10))})
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0.1, qh=0.9, isabs=True, f_agg="mean")
        ),
        pl.DataFrame({"value": [1.0]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0.1, qh=0.9, isabs=False, f_agg="std")
        ),
        pl.DataFrame({"value": [0.0]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(
                pl.col("value"), ql=0.15, qh=0.18, isabs=True, f_agg="mean"
            )
        ),
        pl.DataFrame({"value": [0.0]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0.1, qh=0.9, isabs=True, f_agg="std")
        ),
        pl.DataFrame({"value": [0.0]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0.1, qh=0.9, isabs=False, f_agg="mean")
        ),
        pl.DataFrame({"value": [1.0]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(
                pl.col("value"), ql=0.15, qh=0.18, isabs=False, f_agg="mean"
            )
        ),
        pl.DataFrame({"value": [0.0]}),
    )

    df = pl.DataFrame({"value": [0, 1, 0, 0, 0]})
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0, qh=1, isabs=True, f_agg="mean")
        ),
        pl.DataFrame({"value": [0.5]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0.1, qh=1, isabs=True, f_agg="mean")
        ),
        pl.DataFrame({"value": [0.5]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0.1, qh=0.6, isabs=True, f_agg="mean")
        ),
        pl.DataFrame({"value": [0.0]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0, qh=1, isabs=False, f_agg="mean")
        ),
        pl.DataFrame({"value": [0.0]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0.1, qh=1, isabs=False, f_agg="mean")
        ),
        pl.DataFrame({"value": [0.0]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0.1, qh=0.6, isabs=False, f_agg="mean")
        ),
        pl.DataFrame({"value": [0.0]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0, qh=1, isabs=True, f_agg="std")
        ),
        pl.DataFrame({"value": [0.5]}),
    )

    df = pl.DataFrame({"value": [0, 1, -9, 0, 0]})
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0, qh=1, isabs=True, f_agg="mean")
        ),
        pl.DataFrame({"value": [5.0]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0.1, qh=1, isabs=True, f_agg="mean")
        ),
        pl.DataFrame({"value": [0.5]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0, qh=1, isabs=False, f_agg="mean")
        ),
        pl.DataFrame({"value": [0.0]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0.1, qh=1, isabs=False, f_agg="mean")
        ),
        pl.DataFrame({"value": [0.5]}),
    )

    df = pl.DataFrame({"value": [0, 1, -9, 0, 0, 1, 0]})
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0.1, qh=1, isabs=True, f_agg="mean")
        ),
        pl.DataFrame({"value": [0.75]}),
    )
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0.1, qh=1, isabs=False, f_agg="mean")
        ),
        pl.DataFrame({"value": [0.25]}),
    )

    df = pl.DataFrame({"value": [0, 1, 0, 1, 0]})
    assert_frame_equal(
        df.select(
            change_quantiles(pl.col("value"), ql=0, qh=1, isabs=False, f_agg="std")
        ),
        pl.DataFrame({"value": [1.0]}),
    )


def test_mean_abs_change():
    df = pl.DataFrame({"value": [-2, 2, 5]})
    assert_frame_equal(
        df.select(mean_abs_change(pl.col("value"))), pl.DataFrame({"value": 3.5})
    )
    df = pl.DataFrame({"value": [1, 2, -1]})
    assert_frame_equal(
        df.select(mean_abs_change(pl.col("value"))), pl.DataFrame({"value": [2.0]})
    )
