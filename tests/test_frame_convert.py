from __future__ import annotations

import subprocess
import sys

import narwhals as nw
import narwhals.stable.v2 as nw_v2
import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from functime._frame import coerce_target_series, is_frame_like, to_polars_lazy


def _sample_pl() -> pl.DataFrame:
    return pl.DataFrame(
        {"id": ["a", "a", "b"], "time": [1, 2, 1], "y": [1.0, 2.0, 3.0]}
    )


def _sample_pd() -> pd.DataFrame:
    return pd.DataFrame(
        {"id": ["a", "a", "b"], "time": [1, 2, 1], "y": [1.0, 2.0, 3.0]}
    )


def test_is_frame_like_filters_containers() -> None:
    pl_df = _sample_pl()
    pd_df = _sample_pd()
    for container in (
        pl_df,
        pl_df.lazy(),
        pd_df,
        nw.from_native(pl_df),
        nw.from_native(pl_df.lazy()),
        nw.from_native(pd_df),
        nw_v2.from_native(pl_df),
        nw_v2.from_native(pl_df.lazy()),
        nw_v2.from_native(pd_df),
    ):
        assert is_frame_like(container)

    for non_frame in (
        np.ones((3, 2)),
        pl.Series("y", [1.0, 2.0]),
        pd.Series([1.0, 2.0]),
        [1, 2, 3],
        {"a": [1, 2]},
    ):
        assert not is_frame_like(non_frame)


def test_to_polars_lazy_containers_and_zero_collect() -> None:
    pl_df = _sample_pl()
    pd_df = _sample_pd()

    for container in (
        pl_df,
        pd_df,
        nw.from_native(pl_df),
        nw_v2.from_native(pl_df),
    ):
        lazy_out = to_polars_lazy(container)
        assert isinstance(lazy_out, pl.LazyFrame)
        assert_frame_equal(lazy_out.collect(), pl_df)

    # Polars LazyFrame returns zero-copy identity without collecting.
    pl_lf = pl_df.lazy()
    assert to_polars_lazy(pl_lf) is pl_lf

    deferred_err_lf = (
        pl_df.with_columns(pl.lit("bad").alias("bad"))
        .lazy()
        .with_columns(pl.col("bad").cast(pl.Int64))
    )
    assert isinstance(to_polars_lazy(deferred_err_lf), pl.LazyFrame)
    assert isinstance(to_polars_lazy(nw.from_native(deferred_err_lf)), pl.LazyFrame)
    assert isinstance(to_polars_lazy(nw_v2.from_native(deferred_err_lf)), pl.LazyFrame)

    with pytest.raises(TypeError, match="DataFrame"):
        to_polars_lazy(np.ones((3, 2)))
    with pytest.raises(TypeError, match="DataFrame"):
        to_polars_lazy(pl.Series("y", [1.0, 2.0]))


def test_pyarrow_containers() -> None:
    pa = pytest.importorskip("pyarrow")
    pl_df = _sample_pl()
    pa_tbl = pl_df.to_arrow()
    assert isinstance(pa_tbl, pa.Table)
    assert is_frame_like(pa_tbl)

    lazy_out = to_polars_lazy(pa_tbl)
    assert isinstance(lazy_out, pl.LazyFrame)
    assert_frame_equal(lazy_out.collect(), pl_df)

    s_pa = coerce_target_series(pa.chunked_array([[7.0, 8.0]]), 2)
    assert s_pa.to_list() == [7.0, 8.0]


def test_to_polars_lazy_pandas_index_handling() -> None:
    dates = pd.date_range("2024-01-01", periods=2)
    multi = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)], names=["id", "time"])
    multi_unnamed = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)])

    rejected = [
        pd.DataFrame({"y": [1.0, 2.0]}, index=dates),
        pd.DataFrame(
            {"y": [1.0, 2.0]},
            index=pd.DatetimeIndex(dates, name="time"),
        ),
        pd.DataFrame({"y": [1.0, 2.0]}, index=multi),
        pd.DataFrame({"y": [1.0, 2.0]}, index=multi_unnamed),
        pd.DataFrame(
            {"y": [1.0, 2.0]},
            index=pd.period_range("2024-01", periods=2, freq="M"),
        ),
        pd.DataFrame(
            {"y": [1.0, 2.0]},
            index=pd.timedelta_range("1D", periods=2),
        ),
        pd.DataFrame({"y": [1.0, 2.0]}, index=pd.RangeIndex(2, name="time")),
        pd.DataFrame({"y": [1.0, 2.0]}, index=pd.Index([1, 2], name="time")),
        pd.DataFrame({"y": [1.0, 2.0]}, index=pd.Index(["a", "b"])),
        nw.from_native(pd.DataFrame({"y": [1.0, 2.0]}, index=dates)),
        nw_v2.from_native(pd.DataFrame({"y": [1.0, 2.0]}, index=dates)),
    ]
    for bad in rejected:
        with pytest.raises(ValueError, match="reset_index"):
            to_polars_lazy(bad)

    base = pd.DataFrame({"time": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    allowed = [
        (base, pl.DataFrame({"time": [1, 2, 3], "y": [1.0, 2.0, 3.0]})),
        (base.iloc[1:], pl.DataFrame({"time": [2, 3], "y": [2.0, 3.0]})),
        (base[base["y"] > 1.0], pl.DataFrame({"time": [2, 3], "y": [2.0, 3.0]})),
        (
            base.sort_values("y", ascending=False),
            pl.DataFrame({"time": [3, 2, 1], "y": [3.0, 2.0, 1.0]}),
        ),
        (
            pd.DataFrame(
                {
                    "time": pd.Series(dtype="int64"),
                    "y": pd.Series(dtype="float64"),
                }
            ),
            pl.DataFrame(
                schema={"time": pl.Int64, "y": pl.Float64},
            ),
        ),
    ]
    for pd_in, expected in allowed:
        assert_frame_equal(to_polars_lazy(pd_in).collect(), expected)


def test_coerce_target_series() -> None:
    s_pl = coerce_target_series(pl.Series("orig", [1.0, 2.0, 3.0]), 3, name="y")
    assert s_pl.name == "y"
    assert s_pl.to_list() == [1.0, 2.0, 3.0]

    s_nw = coerce_target_series(nw.from_native(pl.Series([4, 5]), series_only=True), 2)
    assert s_nw.to_list() == [4, 5]

    s_nw_v2 = coerce_target_series(
        nw_v2.from_native(pl.Series([6, 7]), series_only=True), 2
    )
    assert s_nw_v2.to_list() == [6, 7]

    s_pd = coerce_target_series(pd.Series([10.0, 20.0]), 2)
    assert s_pd.to_list() == [10.0, 20.0]

    s_pl_df = coerce_target_series(pl.DataFrame({"val": [7, 8]}), 2)
    assert s_pl_df.to_list() == [7, 8]

    s_nw_v2_df = coerce_target_series(
        nw_v2.from_native(pl.DataFrame({"val": [11, 12]})), 2
    )
    assert s_nw_v2_df.to_list() == [11, 12]

    s_pd_df = coerce_target_series(pd.DataFrame({"val": [9, 10]}), 2)
    assert s_pd_df.to_list() == [9, 10]

    s_list = coerce_target_series([1.0, 2.0], 2)
    assert s_list.to_list() == [1.0, 2.0]

    with pytest.raises(ValueError, match="1D"):
        coerce_target_series(np.zeros((2, 2)), 2)

    with pytest.raises(ValueError, match="1D"):
        coerce_target_series(np.zeros((2, 1)), 2)

    with pytest.raises(ValueError, match="1-column"):
        coerce_target_series(pl.DataFrame({"a": [1], "b": [2]}), 1)

    with pytest.raises(ValueError, match="does not match"):
        coerce_target_series([1, 2], 3)


def test_frame_module_has_no_sklearn_or_pandas_dependency() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, polars as pl; "
            "from functime._frame import is_frame_like, to_polars_lazy; "
            "df = pl.DataFrame({'time': [1, 2], 'y': [1.0, 2.0]}); "
            "assert is_frame_like(df); "
            "assert isinstance(to_polars_lazy(df), pl.LazyFrame); "
            "assert 'sklearn' not in sys.modules and 'pandas' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
