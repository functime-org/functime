import functools
from typing import Any, List, Optional, Union
from warnings import warn

import polars as pl
from statsforecast import StatsForecast


def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warn(f"{func.__name__} is not implemented in Functime.",
                      category=DeprecationWarning,
                      stacklevel=2)
        return func(*args, **kwargs)

    return wrapper

class statsforecast(StatsForecast):
    def __init__(
            self,
            models: List[Any],
            freq: Union[str, int],
            n_jobs: int = 1,
            df: Union[pl.DataFrame, pl.LazyFrame, None] = None,
            sort_df: bool = True,
            fallback_model: Optional[Any] = None,
            verbose: bool = False
        ):
        # Q: What to do here? Either way it will be converted into numpy, so
        # don't see how "saving memory" is effective here
        if isinstance(df, pl.LazyFrame):
            df: pl.DataFrame = df.collect()
        # Note: I won't convert into DataFrame, so LSP will be screaming an error,
        # but it is fine.
        super().__init__(
            models,
            freq,
            n_jobs,
            df,
            sort_df,
            fallback_model,
            verbose
        )

    def plot(self):
        deprecated(self.plot)

    # A bit questionable on this part
    def save(self):
        deprecated(self.save)

    # A bit questionable on this part
    def load(self):
        deprecated(self.load)
