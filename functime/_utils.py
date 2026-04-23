from __future__ import annotations

import logging
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ParamSpec, TypeVar

    P = ParamSpec("P")
    R = TypeVar("R")

logger = logging.getLogger(__name__)


def warn_is_unstable(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> R:
        logger.warning(
            f"The function {func.__name__} is unstable and untested. Use at your own risk."
        )
        return func(*args, **kwargs)

    return wrapped_func
