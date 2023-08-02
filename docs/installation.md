# Installation

functime is published as [Python package in PyPI](https://pypi.org/project/functime/).
To install the latest functime release, run the following command:
```bash
pip install functime
```

## Serverless Deployment

Currently in closed-beta for `functime` Cloud.

Deploy and train forecasters the moment you call any `.fit` method.
Run the `functime list` CLI command to list all deployed models.
Finally, track data and forecasts usage using `functime usage` CLI command.

![Example CLI usage](img/functime_cli_usage.gif)

You can reuse a deployed model for predictions anywhere using the `stub_id` variable.
Note: the `.from_deployed` model class must be the same as during `.fit`.
```python
forecaster = LightGBM.from_deployed(stub_id)
y_pred = forecaster.predict(fh=3)
```
