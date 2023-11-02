import timeit

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from functime.feature_extractor import FeatureExtractor, lempel_ziv_complexity

if __name__ == "__main__":

    df = pl.DataFrame({
        "a": np.random.random(size=(1_000_000,))
    })

    ans = lempel_ziv_complexity(df["a"], 0.2, True)
    print(ans)

