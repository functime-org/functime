from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

HAS_RUST: bool
PLUGIN_PATH: Path | None
rs_faer_lstsq1: Callable[..., Any] | None

try:
    from functime_rust import PLUGIN_PATH as _PLUGIN_PATH
    from functime_rust import rs_faer_lstsq1 as _rs_faer_lstsq1

    HAS_RUST = True
    PLUGIN_PATH = _PLUGIN_PATH
    rs_faer_lstsq1 = _rs_faer_lstsq1
except ImportError:
    HAS_RUST = False
    PLUGIN_PATH = None
    rs_faer_lstsq1 = None
