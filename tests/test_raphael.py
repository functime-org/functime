import polars as pl
from polars.testing import assert_frame_equal

from functime.feature_extraction.features_raphael import change_quantiles

# @pytest.fixture
# def ts_example():
#     # Entity, time
#     names = ["Carl", "John", "Sarah"]
#     dates = pd.date_range("2000-01-01", periods=24)
#     money = np.array([500, 1000, 300])
#     moneys = np.array([np.arange(len(dates))] * len(names)).T * money
#     return names, dates, moneys

# @pytest.fixture
# def ts_data(ts_example):
#     # Example values
#     names, dates, moneys = ts_example
#     # Prepare test data
#     return pl.DataFrame(
#         {
#             "fruit": names * len(dates),
#             "date": dates.repeat(len(names)),
#             "freshness": moneys.flatten(),
#         }
#     ).sort(by=["fruit", "date"])

# @pytest.mark.parametrize("ql,qh,isabs,f_agg", [(0.0, 0.5, True, "mean"), (0.0, 1.0, False, "std"), (0.5, 0.9, True, "var"), (0.25, 0.75, False, "median")])
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
