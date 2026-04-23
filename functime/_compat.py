from __future__ import annotations

from polars.plugins import register_plugin_function

# Polars >=1.0 RLE struct field names
rle_fields = {"value": "value", "len": "len"}

__all__ = ["register_plugin_function", "rle_fields"]
