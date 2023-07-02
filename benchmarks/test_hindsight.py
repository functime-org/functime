import math
import polars as pl
import pytest
import logging

from functime.cross_validation import train_test_split
from functime.preprocessing import time_to_arange

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)


@pytest.fixture
def behacom_dataset():
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

    data = (
        pl.scan_parquet("data/behacom.parquet")
        .filter(pl.col("user") != 2)
        .with_columns(pl.col(pl.Utf8).cast(pl.Categorical).cast(pl.Int16))
        .groupby_dynamic("timestamp", every="1h", by="user")
        .agg([
            pl.col("current_app"),
            pl.col("penultimate_app"),
            pl.all().exclude(["current_app", "penultimate_app", "timestamp"]).mean()
        ])
        # Session
        .with_columns(date=pl.col("timestamp").dt.date())
        .with_columns(
            session_id=(pl.col(label_col).cast(pl.Utf8) + "__" + pl.col("date")).cast(pl.Categorical).cast(pl.Int32)
        )
        .select([entity_col, period_col, pl.all().exclude([entity_col, period_col])])
        # Defensive sort
        .sort([entity_col, time_col])
        .set_sorted([entity_col, time_col])
        .collect(streaming=True)
    )

    # Train test split (predict the next 3 day windows)
    fh = 3
    X_y_train, X_y_test = data.pipe(train_test_split(test_size=fh))
    X_cols = [entity_col, time_col, pl.all().exclude([entity_col, time_col, label_col])]
    y_cols = [entity_col, time_col, pl.col(label_col)]
    X_train, y_train = X_y_train.select(X_cols), X_y_train.select(y_cols)
    X_test, y_test = X_y_test.select(X_cols), X_y_test.select(y_cols)

    # Log dataset
    logging.info(X_train.head())
    logging.info(y_train.head())
    logging.info(X_test.head())
    logging.info(y_test.head())

    return X_train, X_test, y_train, y_test


@pytest.fixture
def elearn_dataset():
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
    # NOTE: Must lazy before join otherwise memory blows up
    entity_col = "session_id"
    time_col = "index"
    keys = ["session_id", "level_group"]
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
        .join(labels, on=["session_id", "level_group"], how="left")
        # Reorder index
        .select([entity_col, time_col, pl.all().exclude([entity_col, time_col])])
        # NOTE: We must coerce `index` into range: we assume users answer questions in order
        .pipe(time_to_arange(keep_col=True))
        # Sort
        .sort([entity_col, time_col])
        .set_sorted([entity_col, time_col])
        .collect(streaming=True)
    )

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

    # Log dataset
    logging.info(X_train.head())
    logging.info(y_train.head())
    logging.info(X_test.head())
    logging.info(y_test.head())

    return X_train, X_test, y_train, y_test, time_col


@pytest.fixture
def datasets(behacom_dataset, elearn_dataset):
    return {
        "behacom": behacom_dataset,
        "elearn": elearn_dataset
    }
