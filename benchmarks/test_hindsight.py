import math
import polars as pl
import pytest
import logging
import cloudpickle

from base64 import b64encode, b64decode
from collections import defaultdict
from functime.cross_validation import train_test_split
from functime.preprocessing import time_to_arange
from functime_backend.classification import Hindsight

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)


def preview_dataset(
    name: str,
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame
):
    """Log memory usage and first 10 rows given train-split splits."""
    logging.info("🔍 Preview %s dataset", name)
    # Log memory
    logging.info("🔍 y_train mem: %s", f'{y_train.estimated_size("mb"):,.4f} mb')
    logging.info("🔍 X_train mem: %s", f'{X_train.estimated_size("mb"):,.4f} mb')
    logging.info("🔍 y_test mem: %s", f'{y_test.estimated_size("mb"):,.4f} mb')
    logging.info("🔍 X_test mem: %s", f'{X_test.estimated_size("mb"):,.4f} mb')
    # Preview dataset
    logging.info("🔍 X_train preview:\n%s", X_train)
    logging.info("🔍 y_train preview:\n%s", y_train)
    logging.info("🔍 X_test preview:\n%s", X_test)
    logging.info("🔍 y_test preview:\n%s", y_test)


def encode_dataset(X_train, X_test, y_train, y_test):
    return {
        "X_train": b64encode(cloudpickle.dumps(X_train)).decode(),
        "X_test": b64encode(cloudpickle.dumps(X_test)).decode(),
        "y_train": b64encode(cloudpickle.dumps(y_train)).decode(),
        "y_test": b64encode(cloudpickle.dumps(y_test)).decode()
    }


def decode_dataset(dataset):
    X_train = cloudpickle.loads(b64decode(dataset["X_train"]))
    X_test = cloudpickle.loads(b64decode(dataset["X_test"]))
    y_train = cloudpickle.loads(b64decode(dataset["y_train"]))
    y_test = cloudpickle.loads(b64decode(dataset["y_test"]))
    return X_train, X_test, y_train, y_test


# NOTE: We set scope to function to lower memory usage
@pytest.fixture(scope="function")
def behacom_dataset(request):
    """Hourly user laptop behavior dataset.

    Train-test split by `session` (`user`, `date-hour`).
    Classify each test session into one of the users (multiclass problem).

    11 users, ~12,000 dimensions (e.g. RAM usage, CPU usage, mouse location),
    ~5000 timestamps. Relevant to IoT and productivity.

    Note: we drop user 2 as they only have 1 day observed.
    """

    entity_col = "session_id"
    time_col = "timestamp"
    period_col = "date"
    label_col = "user"
    fh = 3  # predict the next 3 day windows

    dataset = request.config.cache.get("dataset/behacom", None)
    if dataset is not None:
        logging.info("🗂️ Loading %r dataset from cache", "behacom")
        X_train, X_test, y_train, y_test = decode_dataset(dataset)
        logging.info("✅ %r dataset loaded", "behacom")
    else:    
        data = (
            pl.scan_parquet("data/behacom.parquet")
            .filter(pl.col("user") != 2)
            .groupby_dynamic("timestamp", every="1h", by="user")
            .agg([
                pl.col("current_app"),
                pl.col("penultimate_app"),
                pl.all().exclude(["current_app", "penultimate_app", "timestamp"]).mean()
            ])
            # NOTE: We ordinal encode app usage sequences
            .with_columns([
                pl.col("current_app").list.join("_").cast(pl.Categorical).cast(pl.Int32),
                pl.col("penultimate_app").list.join("_").cast(pl.Categorical).cast(pl.Int32),
            ])
            # Session
            .with_columns(date=pl.col("timestamp").dt.strftime("%Y%m%d").cast(pl.Categorical).cast(pl.Int32))
            .with_columns(session_id=pl.struct([label_col, "date"]).hash())
            .with_columns(session_id=pl.col("session_id").cast(pl.Utf8).cast(pl.Categorical).cast(pl.Int32))
            .select([entity_col, period_col, pl.all().exclude([entity_col, period_col])])
            # Defensive sort
            .sort([entity_col, time_col])
            .set_sorted([entity_col, time_col])
            .select([
                entity_col,
                pl.col(time_col).dt.strftime("%Y%m%dT%H").cast(pl.Categorical).cast(pl.Int32),
                pl.all().exclude([entity_col, time_col])
            ])
            .collect(streaming=True)
        )

        data.write_parquet(".data/behacom.parquet")

        # Train test split
        X_y_train, X_y_test = data.pipe(train_test_split(test_size=fh, eager=True))
        X_cols = [entity_col, time_col, pl.all().exclude([entity_col, time_col, label_col])]
        y_cols = [entity_col, time_col, pl.col(label_col)]
        X_train, y_train = X_y_train.select(X_cols), X_y_train.select(y_cols)
        X_test, y_test = X_y_test.select(X_cols), X_y_test.select(y_cols)

        # Cache dataset
        cached_dataset = encode_dataset(X_train, X_test, y_train, y_test)
        request.config.cache.set("dataset/behacom", cached_dataset)

    preview_dataset("behacom", X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test, period_col


@pytest.fixture(scope="function")
def elearn_dataset(request):
    """Massive e-learning exam score prediction from Kaggle.

    Train-test-split by `session_id`. Predict questions win / loss (softmax).

    Relevant to education, IoT, and online machine learning.
    Link: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/data
    """
    # Game Walkthrough:
    # https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/384796
    question_to_group = {
        **{i+1: "0-4" for i in range(3)},
        **{i+1: "5-12" for i in range(3, 13)},
        **{i+1: "13-22" for i in range(13, 18)},
    }
    entity_col = "session_id"
    time_col = "index"
    keys = ["session_id", "level_group"]

    dataset = request.config.cache.get("dataset/elearn", None)
    if dataset is not None:
        logging.info("🗂️ Loading %r dataset from cache", "elearn")
        X_train, X_test, y_train, y_test = decode_dataset(dataset)
        logging.info("✅ %r dataset loaded", "elearn")
    else:
        labels = (
            pl.scan_parquet("data/elearn_labels.parquet")
            .with_columns(level_group=pl.col("question_id").map_dict(question_to_group).cast(pl.Categorical).cast(pl.Int8))
            # Presort to speed up join
            .sort(keys)
            .set_sorted(keys)
        )
        data = (
            pl.scan_parquet("data/elearn.parquet")
            # Must ordinal encode string columns
            .with_columns(pl.col(pl.Utf8).cast(pl.Categorical).cast(pl.Int32))
            .select(pl.all().shrink_dtype())
            # Presort to speed up join
            .sort(keys)
            .set_sorted(keys)
            # NOTE: Must lazy join otherwise memory blows up
            .join(labels, on=["session_id", "level_group"], how="left")
            # Reorder index
            .select([entity_col, time_col, pl.all().exclude([entity_col, time_col])])
            # NOTE: We must coerce `index` into range: we assume users answer questions in order
            .pipe(time_to_arange(keep_col=False))
            # Sort
            .sort([entity_col, time_col])
            .set_sorted([entity_col, time_col])
            .collect(streaming=True)
        )

        # Subsample entities and first 6 questions
        n = 4
        data = data.pipe(lambda df: df.filter(pl.col("session_id").is_in(df.get_column("session_id").sample(n))))
        data = data.pipe(lambda df: df.filter(pl.col("question_id") <= 12))

        # 10% test size
        test_size = 0.10
        n_test_samples = int(math.ceil(data.select(pl.col("session_id").n_unique()).item() * test_size))
        test_session_ids = data.get_column("session_id").sample(n_test_samples)

        entity_col = "session_id"
        time_col = "index"
        label_col = "correct"

        # Train test split
        X_y_train = data.filter(~pl.col("session_id").is_in(test_session_ids))
        X_y_test = data.filter(pl.col("session_id").is_in(test_session_ids))
        X_cols = [entity_col, time_col, pl.all().exclude([entity_col, time_col, label_col])]
        y_cols = [entity_col, time_col, pl.col(label_col)]
        X_train, y_train = X_y_train.select(X_cols), X_y_train.select(y_cols)
        X_test, y_test = X_y_test.select(X_cols), X_y_test.select(y_cols)

        preview_dataset("elearn", X_train, X_test, y_train, y_test)

        # Cache dataset
        cached_dataset = encode_dataset(X_train, X_test, y_train, y_test)
        request.config.cache.set("dataset/elearn", cached_dataset)

    return X_train, X_test, y_train, y_test, time_col


def _fit_predict(X_train: pl.DataFrame, X_test: pl.DataFrame, y_train: pl.DataFrame, y_test: pl.DataFrame) -> pl.DataFrame:

    # Set expanding window configs
    time_col = y_test.columns[1]
    start = int(y_test.get_column(time_col).max() * 0.5)
    min_ts = X_train.get_column(time_col).min()
    max_ts = X_train.get_column(time_col).max()

    logging.info("💡 Starting expanding window at step: %s", start)
    logging.info("💡 Train data timestamp min-max: [%s, %s]", min_ts, max_ts)

    # Fit
    logging.info("🚀 Training Hindsight...")
    model = Hindsight(start=start, step=1, random_state=42)
    model.fit(X=X_train, y=y_train)
    logging.info("✅ Model training complete")

    # Predict
    logging.info("🔮 Running predict...")
    labels = model.predict(X=X_test)
    y_pred = y_test.select(y_test.columns[:2]).with_columns(pl.Series(name="label", values=labels))
    logging.info("✅ Prediction complete")
    logging.info("📹 Predicted labels preview:\n%s", y_pred)

    return y_pred


def _score(y_test: pl.DataFrame, y_pred: pl.DataFrame, period_col: str):

    entity_col, time_col = y_test.columns[:2]
    y_test = y_test.rename({y_test.columns[-1]: "true"})
    metrics = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
    }
    results = defaultdict(dict)
    logging.info("💯 Scoring Hindsight...")
    for metric_name, metric in metrics.items():
        scores = metric(
            y_true=y_test.get_column("true").to_numpy(),
            y_pred=y_pred.get_column("label").to_numpy()
        )
        logging.info(scores)
        results[metric_name]["full"] = scores
    
    # Score by entity
    for metric_name, metric in metrics.items():
        y_pred_true = y_pred.join(y_test, on=[entity_col, time_col], how="left")
        scores = (
            y_pred_true
            .groupby(entity_col, maintain_order=True)
            .agg(pl.apply(exprs=["true", "label"], function=lambda args: metric(args[0], args[1])))
        )
        logging.info(scores)
        results[metric_name]["entity"] = scores

    # Score by period
    for metric_name, metric in metrics.items():
        y_pred_true = y_pred.join(y_test, on=[entity_col, time_col], how="left")
        scores = (
            y_pred_true
            .groupby(period_col, maintain_order=True)
            .agg(pl.apply(exprs=["true", "label"], function=lambda args: metric(args[0], args[1])))
        )
        logging.info(scores)
        results[metric_name]["time"] = scores
    logging.info("✅ Scoring complete")

    return results


def test_hindsight_on_behacom(behacom_dataset):
    X_train, X_test, y_train, y_test, period_col = behacom_dataset
    y_pred = _fit_predict(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    results = _score(y_test=y_test, y_pred=y_pred, period_col=period_col)


def test_hindsight_on_elearn(elearn_dataset):
    X_train, X_test, y_train, y_test, period_col = elearn_dataset
    y_pred = _fit_predict(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    results = _score(y_test=y_test, y_pred=y_pred, period_col=period_col)
