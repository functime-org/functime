# Installation

functime is published as [Python package in PyPI](https://pypi.org/project/functime/).
To install the latest functime release, run the following command:

```bash
pip install functime
```

## Extras

`functime` comes with extra options. For example, to install `functime` with lightgbm and plotting features:

```bash
pip install "functime[lgb,plot]"
```

- `ann`: To use `ann` (approximate nearest neighbors) forecaster. This leverages the excellent [lancedb](https://lancedb.github.io/lancedb/).
- `cat`: To use `catboost` forecaster through [`catboost`](https://catboost.ai/).
- `xgb`: To use `xgboost` forecaster through [`xgboost`](https://xgboost.readthedocs.io/en/stable/).
- `lgb`: To use `lightgbm` forecaster. `lightgbm` requires compilation (i.e., a compiler like `cmake` and `libomp`), so please refer to the [installation](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#installation-guide) instructions.
- `plot`: To use plotting functions. Powered by [`plotly`](https://plotly.com/python/).
