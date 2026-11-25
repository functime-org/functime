from __future__ import annotations

from pathlib import Path

from functime_rust import _functime_rust
from functime_rust._functime_rust import rs_faer_lstsq1

# PEP 660 editable installs may place the compiled extension outside the source tree.
PLUGIN_PATH = (
    Path(_functime_rust.__file__).parent
    if getattr(_functime_rust, "__file__", None)
    else Path(__file__).parent
)

__all__ = ["PLUGIN_PATH", "rs_faer_lstsq1"]
