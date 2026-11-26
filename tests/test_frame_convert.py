from __future__ import annotations

import inspect
import subprocess
import sys

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal

from functime._frame import coerce_target_series, is_frame_like, to_polars_lazy


def _sample_pl() -> pl.DataFrame:
    return pl.DataFrame(
        {"id": ["a", "a", "b"], "time": [1, 2, 1], "y": [1.0, 2.0, 3.0]}
    )


def test_is_frame_like_filters_containers() -> None:
    pl_df = _sample_pl()
    for container in (
        pl_df,
        pl_df.lazy(),
        pl_df.to_pandas(),
        pl_df.to_arrow(),
        nw.from_native(pl_df),
        nw.from_native(pl_df.lazy()),
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
    pd_df = pl_df.to_pandas()
    pa_tbl = pl_df.to_arrow()
    assert isinstance(pa_tbl, pa.Table)

    for container in (pl_df, pd_df, pa_tbl, nw.from_native(pl_df)):
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
    nw_deferred_lf = nw.from_native(deferred_err_lf)
    assert isinstance(to_polars_lazy(nw_deferred_lf), pl.LazyFrame)

    with pytest.raises(TypeError, match="DataFrame"):
        to_polars_lazy(np.ones((3, 2)))
    with pytest.raises(TypeError, match="DataFrame"):
        to_polars_lazy(pl.Series("y", [1.0, 2.0]))


def test_coerce_target_series() -> None:
    s_pl = coerce_target_series(pl.Series("orig", [1.0, 2.0, 3.0]), 3, name="y")
    assert s_pl.name == "y"
    assert s_pl.to_list() == [1.0, 2.0, 3.0]

    s_nw = coerce_target_series(nw.from_native(pl.Series([4, 5]), series_only=True), 2)
    assert s_nw.to_list() == [4, 5]

    s_pd = coerce_target_series(pd.Series([10.0, 20.0]), 2)
    assert s_pd.to_list() == [10.0, 20.0]

    s_pa = coerce_target_series(pa.chunked_array([[7.0, 8.0]]), 2)
    assert s_pa.to_list() == [7.0, 8.0]

    s_pl_df = coerce_target_series(pl.DataFrame({"val": [7, 8]}), 2)
    assert s_pl_df.to_list() == [7, 8]

    s_pd_df = coerce_target_series(pd.DataFrame({"val": [9, 10]}), 2)
    assert s_pd_df.to_list() == [9, 10]

    s_list = coerce_target_series([1.0, 2.0], 2)
    assert s_list.to_list() == [1.0, 2.0]

    with pytest.raises(ValueError, match="1D"):
        coerce_target_series(np.zeros((2, 2)), 2)

    with pytest.raises(ValueError, match="1-column"):
        coerce_target_series(pl.DataFrame({"a": [1], "b": [2]}), 1)

    with pytest.raises(ValueError, match="does not match"):
        coerce_target_series([1, 2], 3)


def test_frame_module_has_no_sklearn_or_pandas_dependency() -> None:
    assert "functime._frame" in sys.modules
    source = inspect.getsource(sys.modules["functime._frame"])
    assert "sklearn" not in source
    assert "pandas" not in source

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, functime._frame; "
            "assert 'sklearn' not in sys.modules and 'pandas' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
