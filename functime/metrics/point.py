import numpy as np
import polars as pl

from functime.base import metric


def _score(y_true, y_pred, formula: pl.Expr, alias: str):
    y_true = y_true.rename({y_true.columns[-1]: "actual"})
    y_pred = y_pred.rename({y_pred.columns[-1]: "pred"})
    entity_col, time_col = y_true.columns[:2]
    scores = (
        y_true.join(y_pred, on=[entity_col, time_col], how="left")
        .group_by(entity_col)
        .agg(formula.alias(alias))
    )
    return scores


@metric
def mae(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
    """Return mean absolute error (MAE).

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.

    Returns
    -------
    scores : pl.DataFrame
        Score per series.
    """
    abs_error = (pl.col("pred") - pl.col("actual")).abs()
    return _score(y_true, y_pred, abs_error.mean(), "mae")


@metric
def mfe(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
    """Return mean forecast error (MFE) AKA bias.

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.

    Returns
    -------
    scores : pl.DataFrame
        Score per series.
    """
    error = pl.col("pred") - pl.col("actual")
    return _score(y_true, y_pred, error.mean(), "bias")


@metric
def mape(y_true: pl.DataFrame, y_pred: pl.DataFrame):
    """Return mean absolute percentage error (MAPE).

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.

    Returns
    -------
    scores : pl.DataFrame
        Score per series.
    """
    pct_error = (pl.col("actual") - pl.col("pred")).abs() / np.abs(pl.col("actual"))
    return _score(y_true, y_pred, pct_error.mean(), "mape")


@metric
def mse(y_true: pl.DataFrame, y_pred: pl.DataFrame):
    """Return mean squared error (MSE).

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.

    Returns
    -------
    scores : pl.DataFrame
        Score per series.
    """
    squared_error = (pl.col("pred") - pl.col("actual")) ** 2
    return _score(y_true, y_pred, squared_error.mean(), "mse")


@metric
def rmse(y_true: pl.DataFrame, y_pred: pl.DataFrame):
    """Return root mean squared error (RMSE).

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.

    Returns
    -------
    scores : pl.DataFrame
        Score per series.
    """
    squared_error = (pl.col("pred") - pl.col("actual")) ** 2
    return _score(y_true, y_pred, squared_error.mean().sqrt(), "rmse")


@metric
def smape(y_true: pl.DataFrame, y_pred: pl.DataFrame):
    """Return symmetric mean absolute percentage error (sMAPE).

    Use third version of SMAPE formula from https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error to deal with zero division error

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.

    Returns
    -------
    scores : pl.DataFrame
        Score per series.
    """
    diff = (pl.col("pred") - pl.col("actual")).abs().sum()
    total = (pl.col("pred") + pl.col("actual")).sum()
    pct_error = diff / total
    return _score(y_true, y_pred, pct_error, "smape")


@metric
def smape_original(y_true: pl.DataFrame, y_pred: pl.DataFrame):
    """Return symmetric mean absolute percentage error (sMAPE).

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.

    Returns
    -------
    scores : pl.DataFrame
        Score per series.
    """
    num = 2 * (pl.col("pred") - pl.col("actual")).abs()
    denom = 0.0001 + pl.col("actual").abs() + pl.col("pred").abs()
    pct_error = (100 / pl.col("pred").len()) * (num / denom).sum()
    return _score(y_true, y_pred, pct_error, "smape_original")


@metric
def mase(
    y_true: pl.DataFrame, y_pred: pl.DataFrame, y_train: pl.DataFrame, sp: int = 1
):
    """Return mean absolute scaled error (MASE).

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.
    y_train : pl.DataFrame
        Observed training values.

    Returns
    -------
    scores : pl.DataFrame
        Score per series.
    """
    mae_scores = mae(y_true, y_pred)
    naive_mae_scores = (
        y_train.rename({y_train.columns[-1]: "naive"})
        .group_by(y_train.columns[0])
        .agg((pl.col("naive") - pl.col("naive").shift(sp)).abs().mean())
    )
    entity_col = y_true.columns[0]
    scores = (
        mae_scores.lazy()
        .join(naive_mae_scores.lazy(), on=entity_col, how="left")
        .select([entity_col, (pl.col("mae") / pl.col("naive")).alias("mase")])
        .collect(streaming=True)
    )
    return scores


@metric
def rmsse(
    y_true: pl.DataFrame, y_pred: pl.DataFrame, y_train: pl.DataFrame, sp: int = 1
):
    """Return root mean squared scaled error (RMSSE).

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.
    y_train : pl.DataFrame
        Observed training values.

    Returns
    -------
    scores : pl.DataFrame
        Score per series.
    """
    mse_scores = mse(y_true, y_pred)
    naive_mse_scores = (
        y_train.rename({y_train.columns[-1]: "naive"})
        .group_by(y_train.columns[0])
        .agg(((pl.col("naive") - pl.col("naive").shift(sp)) ** 2).mean())
    )
    entity_col = y_true.columns[0]
    scores = (
        mse_scores.lazy()
        .join(naive_mse_scores.lazy(), on=entity_col, how="left")
        .select([entity_col, (pl.col("mse") / pl.col("naive")).sqrt().alias("rmsse")])
        .collect(streaming=True)
    )
    return scores


@metric
def overforecast(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
    """Return total overforecast.

    Overforecast (positive forecast bias) is the difference between actual and predicted for predicted values greater than actual.

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.

    Returns
    -------
    scores : pl.DataFrame
        Score per series.
    """
    overforecast = pl.col("pred").filter(pl.col("pred") > pl.col("actual")).sum()
    return _score(y_true, y_pred, overforecast, "overforecast")


@metric
def underforecast(y_true: pl.DataFrame, y_pred: pl.DataFrame) -> pl.DataFrame:
    """Return total underforecast.

    Underforecast (negative forecast bias) is the difference between actual and predicted for predicted values less than actual.

    Parameters
    ----------
    y_true : pl.DataFrame
        Ground truth (correct) target values.
    y_pred : pl.DataFrame
        Predicted values.

    Returns
    -------
    scores : pl.DataFrame
        Score per series.
    """
    underforecast = pl.col("pred").filter(pl.col("pred") < pl.col("actual")).sum()
    return _score(y_true, y_pred, underforecast, "underforecast")
