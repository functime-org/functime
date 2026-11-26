from __future__ import annotations

import functools
import re
from collections.abc import Sequence
from html.parser import HTMLParser
from typing import Any

import polars as pl
import pytest
from sklearn import config_context
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import estimator_html_repr

import functime.sklearn._base as _base
from functime._frame import TemporalFrame
from functime.sklearn import BaseTemporalTransformer
from functime.sklearn._base import _DataFrameTransformerMixin

_HEX_ADDR_RE = re.compile(r"0x[0-9a-fA-F]{4,}")
_VOID_ELEMENTS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


class _TagBalanceParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.stack: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if self._skip_depth > 0:
            return
        if tag_lower in ("style", "script"):
            self._skip_depth += 1
        if tag_lower not in _VOID_ELEMENTS:
            self.stack.append(tag_lower)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return None

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if self._skip_depth > 0:
            if tag_lower in ("style", "script"):
                self._skip_depth -= 1
                assert self.stack and self.stack[-1] == tag_lower
                self.stack.pop()
            return
        if tag_lower in _VOID_ELEMENTS:
            return
        assert self.stack, f"Unexpected closing tag </{tag_lower}>"
        top = self.stack.pop()
        assert top == tag_lower, f"Mismatched closing tag </{tag_lower}> for <{top}>"


def _assert_balanced_html(markup: str) -> None:
    parser = _TagBalanceParser()
    parser.feed(markup)
    parser.close()
    assert not parser.stack, f"Unclosed HTML tags: {parser.stack}"


def _sample_fn(x: Any, mult: int = 1) -> Any:
    return x * mult


class _CallableObj:
    def __call__(self, x: Any) -> Any:
        return x


class _DocumentedTemporalTransformer(BaseTemporalTransformer):
    """Documented test transformer for HTML representation checks.

    Parameters
    ----------
    time_col : str, default="time"
        Name of the timestamp column.
    entity_col : str or sequence of str or None, default=None
        Optional entity identifier column(s).
    expr : pl.Expr or None, default=None
        Optional Polars expression applied to values.
    fn : callable or None, default=None
        Optional callable parameter.
    """

    __module__ = "functime.sklearn.preprocessing._dummy"

    def __init__(
        self,
        time_col: str = "time",
        entity_col: str | Sequence[str] | None = None,
        expr: pl.Expr | None = None,
        fn: Any = None,
    ) -> None:
        super().__init__(time_col=time_col, entity_col=entity_col)
        self.expr = expr
        self.fn = fn

    def _fit_frame(self, frame: TemporalFrame, y: Any = None) -> None:
        self._state_ = pl.DataFrame({"col": ["a", "b"], "val": [1.0, 2.0]})
        self.feature_names_out_ = tuple(frame.columns)

    def _transform_frame(self, frame: TemporalFrame) -> TemporalFrame:
        return frame


class _TopLevelDocEstimator(_DataFrameTransformerMixin):
    """Top-level functime estimator for doc link fallback check."""

    __module__ = "functime._top"

    def _fit_df(self, df: pl.DataFrame, y: Any = None) -> None:
        self.feature_names_out_ = tuple(df.columns)

    def _transform_df(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


class _DefaultExprEstimator(_DataFrameTransformerMixin):
    """Estimator whose default parameter is a Polars expression."""

    __module__ = "functime.sklearn.preprocessing._dummy"

    def __init__(self, expr: pl.Expr = pl.col("y")) -> None:
        self.expr = expr

    def _fit_df(self, df: pl.DataFrame, y: Any = None) -> None:
        self.feature_names_out_ = tuple(df.columns)

    def _transform_df(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


def _sample_df() -> pl.DataFrame:
    return pl.DataFrame({"id": ["a", "a"], "time": [1, 2], "y": [1.0, 2.0]})


def test_unfitted_and_fitted_html_repr_and_display_modes() -> None:
    est = _DocumentedTemporalTransformer(time_col="time", entity_col="id")

    with config_context(display="diagram"):
        html_unfitted = est._repr_html_()
        bundle_diagram = est._repr_mimebundle_()
        assert "text/html" in bundle_diagram
        assert "text/plain" in bundle_diagram
        assert "sk-estimator" in html_unfitted
        assert "sk-estimator-doc-link fitted" not in html_unfitted
        _assert_balanced_html(html_unfitted)

        est.fit(_sample_df())
        html_fitted = est._repr_html_()
        assert "fitted" in html_fitted
        assert "sk-estimator-doc-link fitted" in html_fitted
        _assert_balanced_html(html_fitted)

    with config_context(display="text"):
        with pytest.raises(AttributeError, match="_repr_html_"):
            _ = est._repr_html_
        bundle_text = est._repr_mimebundle_()
        assert "text/html" not in bundle_text
        assert "text/plain" in bundle_text


def test_doc_link_urls_and_numpy_parameter_tooltips() -> None:
    est = _DocumentedTemporalTransformer(time_col="time", entity_col="id")
    assert est._doc_link_module == "functime"
    expected_url = (
        "https://functime-org.github.io/functime/api-reference/"
        "sklearn-preprocessing/#functime.sklearn.preprocessing._DocumentedTemporalTransformer"
    )
    assert est._get_doc_link() == expected_url

    with config_context(display="diagram"):
        html_out = estimator_html_repr(est)
        assert expected_url in html_out
        hrefs = re.findall(r'href="([^"]+)"', html_out)
        assert hrefs and all("scikit-learn.org" not in href for href in hrefs)
        assert "param-doc-link" in html_out
        assert "param-doc-description" in html_out
        assert "Name of the timestamp column." in html_out

    top = _TopLevelDocEstimator()
    assert top._get_doc_link() == (
        "https://functime-org.github.io/functime/api-reference/"
        "index/#functime._TopLevelDocEstimator"
    )


def test_polars_expr_and_callable_params_repr_no_hex_addresses() -> None:
    partial_fn = functools.partial(_sample_fn, mult=3)
    cases = [
        _DocumentedTemporalTransformer(expr=pl.col("y") * 2, fn=partial_fn),
        _DocumentedTemporalTransformer(expr=pl.col("y").abs(), fn=_sample_fn),
        _DocumentedTemporalTransformer(fn=lambda x: x),
        _DocumentedTemporalTransformer(fn=_CallableObj()),
        _DefaultExprEstimator(),
        _DefaultExprEstimator(expr=(pl.col("y") + 1)),  # type: ignore[arg-type]
    ]
    for est in cases:
        plain = repr(est)
        assert not _HEX_ADDR_RE.search(plain), f"Unexpected hex address in {plain!r}"
        with config_context(display="diagram"):
            html_out = estimator_html_repr(est)
            assert not _HEX_ADDR_RE.search(html_out), (
                f"Unexpected hex address in HTML repr for {est!r}"
            )
            _assert_balanced_html(html_out)

    # Tuple of Polars expressions & DataFrame/Series params.
    est_containers = _DocumentedTemporalTransformer(
        expr=(pl.col("y"), pl.col("y").shift(1)),  # type: ignore[arg-type]
        fn={"df": pl.DataFrame({"a": [1, 2]}), "s": pl.Series("v", [3, 4])},
    )
    plain_containers = repr(est_containers)
    assert not _HEX_ADDR_RE.search(plain_containers)
    with config_context(display="diagram"):
        html_containers = estimator_html_repr(est_containers)
        assert not _HEX_ADDR_RE.search(html_containers)
        _assert_balanced_html(html_containers)


def test_pipeline_column_transformer_and_feature_union_html_repr() -> None:
    df = _sample_df()
    pipe = Pipeline(
        [
            ("panel", _DocumentedTemporalTransformer(time_col="time", entity_col="id")),
            ("scale", StandardScaler()),
        ]
    )
    ct = ColumnTransformer(
        [
            (
                "panel",
                _DocumentedTemporalTransformer(time_col="time", entity_col="id"),
                ["id", "time", "y"],
            )
        ]
    )
    fu = FeatureUnion(
        [("panel", _DocumentedTemporalTransformer(time_col="time", entity_col="id"))]
    )

    with config_context(display="diagram"):
        for composite in (pipe, ct, fu):
            html_out = estimator_html_repr(composite)
            assert "functime-org.github.io" in html_out
            assert "_DocumentedTemporalTransformer" in html_out
            _assert_balanced_html(html_out)

        pipe_fitted = Pipeline(
            [
                (
                    "panel",
                    _DocumentedTemporalTransformer(
                        time_col="time", entity_col="id", expr=pl.col("y") * 2
                    ),
                ),
            ]
        ).fit(df)
        html_fitted = estimator_html_repr(pipe_fitted)
        diagram_html = html_fitted[html_fitted.index('class="sk-container"') :]
        assert not _HEX_ADDR_RE.search(diagram_html)
        assert "pl.Expr(" in diagram_html
        _assert_balanced_html(html_fitted)


def test_state_hygiene_and_xss_escaping() -> None:
    est = _DocumentedTemporalTransformer(
        time_col="<script>alert(1)</script>",
        expr=pl.col("<script>alert(2)</script>"),
    )
    est.fit(
        pl.DataFrame(
            {
                "<script>alert(1)</script>": [1, 2],
                "<script>alert(2)</script>": [10.0, 20.0],
            }
        )
    )
    with config_context(display="diagram"):
        html_out = est._repr_html_()
        for box_char in ("┌", "└", "╞"):
            assert box_char not in html_out
        assert "<script>alert(1)</script>" not in html_out
        assert "<script>alert(2)</script>" not in html_out
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out
        _assert_balanced_html(html_out)


def test_sklearn_1_6_params_dict_none_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_base, "_ParamsDict", None)
    est = _DocumentedTemporalTransformer(
        time_col="time",
        entity_col="id",
        expr=pl.col("y") * 2,
        fn=functools.partial(_sample_fn, mult=5),
    )
    assert not hasattr(est, "_get_params_html")

    with config_context(display="diagram"):
        html_repr = estimator_html_repr(est)
        html_method = est._repr_html_()
        for html_out in (html_repr, html_method):
            assert "functime-org.github.io" in html_out
            assert not _HEX_ADDR_RE.search(html_out)
            assert "<pre>" in html_out
            _assert_balanced_html(html_out)
