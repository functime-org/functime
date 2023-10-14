# Installation

functime is published as [Python package in PyPI](https://pypi.org/project/functime/).
To install the latest functime release, run the following command:

```bash
pip install functime
```

## Extras

`functime` comes with extra options. For example, to install `functime` with large-language model (LLM) and lightgbm features:

```bash
pip install "functime[llm,lgb]"
```

- `ann`: To use `ann` (approximate nearest neighbors) forecaster
- `cat`: To use `catboost` forecaster
- `xgb`: To use `xgboost` forecaster
- `lgb`: To use `lightgbm` forecaster
- `llm`: To use the LLM-powered forecast analyst
- `plot`: To use plotting functions
