from __future__ import annotations

import inspect
import subprocess
import sys
from collections.abc import Sequence
from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from functime._frame import (
    SortState,
    TemporalFrame,
    clean_float_expr,
    detect_sort_state,
    normalize_entity_cols,
    temporal_transform,
    validate_temporal_params,
)


def test_normalize_entity_cols_and_validate_temporal_params() -> None:
    assert normalize_entity_cols(None) == ()
    assert normalize_entity_cols("entity") == ("entity",)
    assert normalize_entity_cols(["store", "sku"]) == ("store", "sku")

    assert validate_temporal_params("time", None) == ()
    assert validate_temporal_params("time", ["store", "sku"]) == ("store", "sku")

    with pytest.raises(ValueError, match="empty string"):
        normalize_entity_cols("")
    with pytest.raises(ValueError, match="empty"):
        normalize_entity_cols([])
    with pytest.raises(TypeError, match="bytes"):
        normalize_entity_cols(b"id")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="non-empty strings"):
        normalize_entity_cols(["id", ""])
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_entity_cols(["id", "id"])

    for bad_time in ("", 123):
        with pytest.raises(ValueError, match="time_col"):
            validate_temporal_params(bad_time)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cannot also appear"):
        validate_temporal_params("time", ["id", "time"])


def test_detect_sort_state_variants() -> None:
    assert (
        detect_sort_state(pl.DataFrame({"t": pl.Series([], dtype=pl.Int64)}), (), "t")
        == SortState.ENTITY_TIME_SORTED
    )
    assert (
        detect_sort_state(pl.DataFrame({"t": [10]}), (), "t")
        == SortState.ENTITY_TIME_SORTED
    )

    uni_sorted = pl.DataFrame({"t": [1, 2, 3]}).sort("t")
    assert detect_sort_state(uni_sorted, (), "t") == SortState.ENTITY_TIME_SORTED

    uni_unsorted = pl.DataFrame({"t": [2, 1, 3]})
    assert detect_sort_state(uni_unsorted, (), "t") == SortState.UNKNOWN
    assert (
        detect_sort_state(uni_unsorted, (), "t", check_scan=True) == SortState.UNKNOWN
    )

    # [time, entity] interleaved sorted panel sets SORTED_ASC on time_col -> CHRONOLOGICAL in O(1).
    te_sorted = pl.DataFrame(
        {"id": ["b", "a", "b", "a"], "t": [1, 1, 2, 2], "y": [1.0, 2.0, 3.0, 4.0]}
    ).sort(["t", "id"])
    assert detect_sort_state(te_sorted, ("id",), "t") == SortState.CHRONOLOGICAL

    # [entity, time] sorted panel sets SORTED_ASC on entity -> O(1) without scan, ENTITY_TIME_SORTED with scan.
    et_sorted = pl.DataFrame(
        {"id": ["b", "a", "b", "a"], "t": [2, 2, 1, 1], "y": [1.0, 2.0, 3.0, 4.0]}
    ).sort(["id", "t"])
    assert detect_sort_state(et_sorted, ("id",), "t") == SortState.UNKNOWN
    assert (
        detect_sort_state(et_sorted, ("id",), "t", check_scan=True)
        == SortState.ENTITY_TIME_SORTED
    )

    # Entity sorted, bad time order inside entity.
    bad_time = pl.DataFrame({"id": ["a", "a", "b", "b"], "t": [2, 1, 1, 2]}).sort(
        "id", maintain_order=True
    )
    assert (
        detect_sort_state(bad_time, ("id",), "t", check_scan=True) == SortState.UNKNOWN
    )

    # Multi-column entity fallback.
    multi_sorted = pl.DataFrame(
        {
            "store": ["s2", "s1", "s1", "s2"],
            "sku": ["b", "a", "a", "b"],
            "t": [2, 2, 1, 1],
        }
    ).sort(["store", "sku", "t"])
    assert (
        detect_sort_state(multi_sorted, ("store", "sku"), "t", check_scan=True)
        == SortState.CHRONOLOGICAL
    )


def test_temporal_frame_parse_always_lazy_idempotence_and_errors() -> None:
    df = pl.DataFrame(
        {
            "store": ["s1", "s1"],
            "sku": ["a", "b"],
            "time": [1, 2],
            "y": [10.0, 20.0],
            "x": [1.0, 2.0],
        }
    ).sort("time")

    tf = TemporalFrame.parse(df, time_col="time", entity_col=["store", "sku"])
    assert isinstance(tf.lf, pl.LazyFrame)
    assert tf.is_panel
    assert tf.entity_cols == ("store", "sku")
    assert tf.target_cols == ("y", "x")
    assert tf.columns == ["store", "sku", "time", "y", "x"]
    assert tf.sort_state == SortState.CHRONOLOGICAL

    # Eager single-entity [entity, time]-sorted frame upgrades to ENTITY_TIME_SORTED at parse.
    et_single = df.sort(["store", "time"])
    tf_et = TemporalFrame.parse(et_single, time_col="time", entity_col="store")
    assert tf_et.sort_state == SortState.ENTITY_TIME_SORTED

    # Re-parsing TemporalFrame is idempotent and preserves elevated sort_state.
    tf_sorted = tf.sort_entity_time()
    assert tf_sorted.sort_state == SortState.ENTITY_TIME_SORTED
    reparsed = TemporalFrame.parse(
        tf_sorted, time_col="time", entity_col=["store", "sku"], columns=["y"]
    )
    assert reparsed.target_cols == ("y",)
    assert reparsed.sort_state == SortState.ENTITY_TIME_SORTED

    # Lazy parse inspects schema without collecting or evaluating expressions.
    lf_deferred_error = (
        df.with_columns(pl.lit("not_an_int").alias("bad"))
        .lazy()
        .with_columns(pl.col("bad").cast(pl.Int64))
    )
    tf_deferred = TemporalFrame.parse(
        lf_deferred_error, time_col="time", entity_col="store"
    )
    assert isinstance(tf_deferred.lf, pl.LazyFrame)
    assert tf_deferred.sort_state == SortState.UNKNOWN

    # Schema & null validation.
    with pytest.raises(ValueError, match="Missing required"):
        TemporalFrame.parse(df, time_col="missing")
    with pytest.raises(ValueError, match="Specified column"):
        TemporalFrame.parse(df, time_col="time", columns=["nope"])
    with pytest.raises(ValueError, match="0 rows"):
        TemporalFrame.parse(df.head(0), time_col="time", entity_col="store")

    df_null_t = df.with_columns(pl.Series("time", [1, None]))
    with pytest.raises(ValueError, match="null"):
        TemporalFrame.parse(df_null_t, time_col="time", entity_col="store")
    TemporalFrame.parse(
        df_null_t, time_col="time", entity_col="store", allow_null_time=True
    )

    df_null_e = df.with_columns(pl.Series("store", ["s1", None]))
    with pytest.raises(ValueError, match="null"):
        TemporalFrame.parse(df_null_e, time_col="time", entity_col="store")
    TemporalFrame.parse(
        df_null_e, time_col="time", entity_col="store", allow_null_entity=True
    )


def test_over_and_with_exprs_parity_and_row_order_on_shuffled() -> None:
    df_panel = pl.DataFrame(
        {
            "entity": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "time": [
                date(2024, 1, 3),
                date(2024, 1, 1),
                date(2024, 1, 4),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 4),
                date(2024, 1, 1),
                date(2024, 1, 3),
            ],
            "y": [30.0, 10.0, None, 20.0, 200.0, 400.0, 100.0, None],
        }
    )
    exprs = [
        pl.col("y").shift(1).alias("shift1"),
        pl.col("y").diff(1).alias("diff1"),
        pl.col("y").forward_fill().alias("ffill"),
        pl.col("y").ewm_mean(span=2).alias("ewm"),
        pl.col("y").rolling_mean(window_size=2).alias("rolling_int"),
        pl.col("y")
        .mean()
        .rolling(index_column="time", period="2d")
        .alias("rolling_2d"),
    ]

    ref_panel = (
        df_panel.with_row_index("__idx")
        .sort(["entity", "time"], maintain_order=True)
        .with_columns([e.over("entity") for e in exprs])
        .sort("__idx")
        .drop("__idx")
    )
    out_panel = (
        TemporalFrame.parse(df_panel, time_col="time", entity_col="entity")
        .with_exprs(exprs, chronological=True)
        .collect()
    )
    assert_frame_equal(out_panel, ref_panel)

    # Same exact parity for shuffled univariate series.
    df_uni = df_panel.filter(pl.col("entity") == "A").drop("entity")
    ref_uni = (
        df_uni.with_row_index("__idx")
        .sort("time", maintain_order=True)
        .with_columns(exprs)
        .sort("__idx")
        .drop("__idx")
    )
    out_uni = (
        TemporalFrame.parse(df_uni, time_col="time")
        .with_exprs(exprs, chronological=True)
        .collect()
    )
    assert_frame_equal(out_uni, ref_uni)


def test_group_agg_chronological_eager_and_lazy_inputs() -> None:
    df = pl.DataFrame(
        {
            "id": ["b", "a", "a", "b"],
            "t": [2, 2, 1, 1],
            "y": [20.0, 2.0, 1.0, 10.0],
        }
    )
    agg_exprs = [pl.col("y").diff(1).last().alias("last_diff")]

    out_from_eager_lf = TemporalFrame.parse(
        df, time_col="t", entity_col="id"
    ).group_agg(agg_exprs, chronological=True)
    out_from_lazy_lf = TemporalFrame.parse(
        df.lazy(), time_col="t", entity_col="id"
    ).group_agg(agg_exprs, chronological=True)
    assert isinstance(out_from_eager_lf, pl.LazyFrame)
    assert isinstance(out_from_lazy_lf, pl.LazyFrame)

    out_eager = out_from_eager_lf.collect().sort("id")
    assert out_eager.get_column("last_diff").to_list() == [1.0, 10.0]
    assert_frame_equal(out_from_lazy_lf.collect().sort("id"), out_eager)

    # Univariate group_agg.
    df_u = df.filter(pl.col("id") == "a").drop("id")
    out_u_lf = TemporalFrame.parse(df_u, time_col="t").group_agg(
        agg_exprs, chronological=True
    )
    assert isinstance(out_u_lf, pl.LazyFrame)
    assert out_u_lf.collect().get_column("last_diff").to_list() == [1.0]


def test_join_state_univariate_panel_dtypes_and_handle_unknown() -> None:
    df = pl.DataFrame({"t": [1, 2, 3], "y": [10.0, 20.0, 30.0]})
    stats_uni = pl.DataFrame(
        {"mean_y": [20.0], "null_stat": pl.Series([None], dtype=pl.Float64)}
    )
    out_uni = TemporalFrame.parse(df, time_col="t").join_state(stats_uni).collect()
    assert out_uni.schema["mean_y"] == pl.Float64
    assert out_uni.schema["null_stat"] == pl.Float64
    assert out_uni.get_column("mean_y").to_list() == [20.0, 20.0, 20.0]

    with pytest.raises(ValueError, match="1-row"):
        TemporalFrame.parse(df, time_col="t").join_state(
            pl.DataFrame({"mean_y": [1.0, 2.0]})
        )

    # Panel join preserves order, String/Categorical/Enum dtypes, and integer widths.
    enum_ab = pl.Enum(["a", "b"])
    df_typed = pl.DataFrame(
        {
            "id": pl.Series(["b", "a", "b"], dtype=enum_ab),
            "t": [2, 1, 1],
            "__ft_join_id_0": [1, 2, 3],
            "y": [1.0, 2.0, 3.0],
        }
    )
    stats_typed = pl.DataFrame(
        {"id": pl.Series(["a", "b"], dtype=pl.String), "offset": [10.0, 20.0]}
    )
    aligned_tf = TemporalFrame.parse(
        df_typed, time_col="t", entity_col="id"
    ).join_state(stats_typed)
    assert isinstance(aligned_tf.lf, pl.LazyFrame)
    aligned = aligned_tf.collect()
    assert aligned.schema["id"] == enum_ab
    assert aligned.columns == ["id", "t", "__ft_join_id_0", "y", "offset"]
    assert aligned.get_column("offset").to_list() == [20.0, 10.0, 20.0]

    df_int = pl.DataFrame(
        {"id": pl.Series([2, 1], dtype=pl.Int32), "t": [1, 1], "y": [1.0, 2.0]}
    )
    stats_int = pl.DataFrame(
        {"id": pl.Series([1, 2], dtype=pl.Int64), "offset": [5.0, 6.0]}
    )
    aligned_int = (
        TemporalFrame.parse(df_int, time_col="t", entity_col="id")
        .join_state(stats_int)
        .collect()
    )
    assert aligned_int.schema["id"] == pl.Int32
    assert aligned_int.get_column("offset").to_list() == [6.0, 5.0]

    # handle_unknown modes work cleanly even when initialized from LazyFrame.
    df_unknown = pl.DataFrame({"id": ["a", "b"], "t": [1, 1], "y": [1.0, 2.0]})
    stats_partial = pl.DataFrame({"id": ["a"], "offset": [10.0]})
    for mode in ("global", "identity", "null"):
        joined = (
            TemporalFrame.parse(df_unknown.lazy(), time_col="t", entity_col="id")
            .join_state(stats_partial, handle_unknown=mode)
            .collect()
        )
        assert joined.get_column("offset").to_list() == [10.0, None]

    with pytest.raises(ValueError, match="unseen entity key"):
        TemporalFrame.parse(
            df_unknown.lazy(), time_col="t", entity_col="id"
        ).join_state(stats_partial, handle_unknown="error")
    with pytest.raises(ValueError, match="handle_unknown"):
        TemporalFrame.parse(df_unknown, time_col="t", entity_col="id").join_state(
            stats_partial,
            handle_unknown="invalid",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="Incompatible dtypes"):
        TemporalFrame.parse(df_unknown, time_col="t", entity_col="id").join_state(
            pl.DataFrame({"id": [1], "offset": [10.0]})
        )


def test_temporal_transform_always_returns_lazyframe_and_preserves_plan() -> None:
    @temporal_transform(chronological=True)
    def diff_op(columns: Sequence[str], *, order: int = 1) -> list[pl.Expr]:
        return [pl.col(c).diff(order).alias(f"{c}_diff_{order}") for c in columns]

    @temporal_transform(chronological=True)
    def lag_op(columns: Sequence[str], *, lag: int = 1) -> list[pl.Expr]:
        return [pl.col(c).shift(lag).alias(f"{c}_lag_{lag}") for c in columns]

    @temporal_transform(chronological=False, needs_time_col=True)
    def year_op(_columns: Sequence[str], *, time_col: str) -> list[pl.Expr]:
        return [pl.col(time_col).dt.year().alias("year")]

    assert diff_op.expr_fn(["y"], order=2)[0].meta.eq(
        pl.col("y").diff(2).alias("y_diff_2")
    )

    df = pl.DataFrame(
        {
            "entity": ["A", "A", "B", "B"],
            "time": [
                date(2024, 1, 2),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 1),
            ],
            "y": [20.0, 10.0, 200.0, 100.0],
        }
    )

    # All inputs (Polars DataFrame, Pandas DataFrame, Polars LazyFrame) always return pl.LazyFrame.
    lf_from_df = diff_op(df, time_col="time", entity_col="entity", order=1)
    assert isinstance(lf_from_df, pl.LazyFrame)
    assert lf_from_df.collect().get_column("y_diff_1").to_list() == [
        10.0,
        None,
        100.0,
        None,
    ]

    lf_from_pd = year_op(df.to_pandas(), time_col="time", entity_col="entity")
    assert isinstance(lf_from_pd, pl.LazyFrame)
    assert lf_from_pd.collect().get_column("year").to_list() == [2024, 2024, 2024, 2024]

    # Chained pipes from eager or lazy frame form one lazy plan with predicate pushdown and zero SORT BY.
    lazy_plan = (
        df.pipe(diff_op, time_col="time", entity_col="entity", order=1)
        .pipe(lag_op, time_col="time", entity_col="entity", lag=1)
        .pipe(year_op, time_col="time", entity_col="entity")
        .filter(pl.col("entity") == "A")
    )
    assert isinstance(lazy_plan, pl.LazyFrame)
    explain_text = lazy_plan.explain()
    assert "SORT BY" not in explain_text
    collected = lazy_plan.collect()
    assert collected.get_column("y_diff_1").to_list() == [10.0, None]
    assert collected.get_column("y_lag_1").to_list() == [10.0, None]


def test_clean_float_expr_and_with_lazy() -> None:
    df = pl.DataFrame({"t": [1, 2], "y": [1.5, float("nan")]}).sort("t")
    cleaned = df.select(clean_float_expr("y"))
    assert cleaned.schema["y"] == pl.Float64
    assert cleaned.get_column("y").to_list() == [1.5, None]

    tf = TemporalFrame.parse(df, time_col="t")
    assert tf.sort_state == SortState.ENTITY_TIME_SORTED
    tf_same_order = tf.with_lazy(tf.lf.with_columns(pl.col("y") * 2))
    assert tf_same_order.sort_state == SortState.ENTITY_TIME_SORTED
    tf_reset = tf.with_lazy(tf.lf.sort("y"), preserves_order=False)
    assert tf_reset.sort_state == SortState.UNKNOWN


def test_frame_module_has_no_sklearn_or_pandas_dependency() -> None:
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
