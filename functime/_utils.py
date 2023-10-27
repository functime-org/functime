import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def UseAtOwnRisk(func: Callable) -> Any:
    def wrapped_func(*args, **kwargs):
        logger.warning(
            f"The function {func.__name__} is unstable and untested. Use at your own risk."
        )
        return func(*args, **kwargs)

    return wrapped_func
