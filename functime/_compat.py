import polars as pl
from typing import Any, Iterable
from pathlib import Path
try:
    from polars.plugins import register_plugin_function
except ImportError:

    def register_plugin_function(*,
        plugin_path: Path | str,
        function_name: str,
        args: 'IntoExpr | Iterable[IntoExpr]',
        kwargs: dict[str, Any] | None = None,
        is_elementwise: bool = False,
        # changes_length: bool = False,
        returns_scalar: bool = False,
        cast_to_supertype: bool = False,
        # input_wildcard_expansion: bool = False,
        # pass_name_to_apply: bool = False,
        ):

        expr = args[0]
        args1 = args[1:]
        expr.register_plugin(
            lib=plugin_path,
            args=args1,
            symbol=function_name,
            is_elementwise=is_elementwise,
            returns_scalar=returns_scalar,
            kwargs=kwargs,
            cast_to_supertypes=cast_to_supertype,
        )