import polars as pl
import polars.selectors as ps
import pytest
import logging
import cloudpickle

from base64 import b64encode, b64decode
from typing import List, Tuple
from functime_backend.classification import Hindsight
from functime.cross_validation import train_test_split

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)


TEST_FRACTION = 0.10
TEST_HORIZON = 3


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
    logging.debug("🔍 X_train preview:\n%s", X_train)
    logging.debug("🔍 y_train preview:\n%s", y_train)
    logging.debug("🔍 X_test preview:\n%s", X_test)
    logging.debug("🔍 y_test preview:\n%s", y_test)
    # Description
    logging.debug("🔍 X_train info:\n%s", X_train.describe())
    logging.debug("🔍 y_train info:\n%s", y_train.describe())
    logging.debug("🔍 X_test info:\n%s", X_test.describe())
    logging.debug("🔍 y_test info:\n%s", y_test.describe())


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


def split_iid_data(
    data: pl.DataFrame,
    label_cols: List[str],
    test_size: float = TEST_FRACTION,
    seed: int = 42
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:

    # Sample session ids
    entity_col = data.columns[0]
    time_col = data.columns[1]
    test_session_ids = data.get_column(entity_col).unique().sample(fraction=test_size, seed=seed)

    # Train test split
    data = data.lazy()
    # Cannot combine 'streaming' with 'common_subplan_elimination'. CSE will be turned off.
    X_y_train, X_y_test = pl.collect_all([
        data.filter(~pl.col(entity_col).is_in(test_session_ids)),
        data.filter(pl.col(entity_col).is_in(test_session_ids))
    ])

    # Split into X, y
    X_cols = [entity_col, time_col, pl.all().exclude([entity_col, time_col, *label_cols])]
    y_cols = [entity_col, time_col, pl.col(label_cols)]

    # Splits
    X_train, y_train = X_y_train.select(X_cols), X_y_train.select(y_cols)
    X_test, y_test = X_y_test.select(X_cols), X_y_test.select(y_cols)

    return X_train, X_test, y_train, y_test


def split_autocorrelated_data(
    data: pl.DataFrame,
    label_cols: List[str],
    test_size: int = TEST_HORIZON,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:

    # Sample session ids
    entity_col = data.columns[0]
    time_col = data.columns[1]
    # Split into X, y
    X_cols = [entity_col, time_col, pl.all().exclude([entity_col, time_col, *label_cols])]
    y_cols = [entity_col, time_col, pl.col(label_cols)]
    # Splits
    X_train, X_test = data.select(X_cols).pipe(train_test_split(test_size=test_size, eager=True))
    y_train, y_test = data.select(y_cols).pipe(train_test_split(test_size=test_size, eager=True))
    # Defensive drop nulls
    X_train = X_train.drop_nulls().unique([entity_col, time_col])
    y_train = y_train.drop_nulls().unique([entity_col, time_col])

    return X_train, X_test, y_train, y_test


def step_from_windows(X: pl.DataFrame, min_windows: int = 10, start_fraction: float = 0.5):
    time_col = X.columns[1]
    min_ts = X.get_column(time_col).min()
    max_ts = X.get_column(time_col).max()
    n_timestamps = len(pl.arange(min_ts, max_ts, step=1, eager=True))
    start = int(n_timestamps * start_fraction)
    full_length = n_timestamps - start
    n_windows = min_windows + (full_length % min_windows)
    step = int(full_length // n_windows)
    return start, step


@pytest.fixture(params=[3, 6, 10], ids=lambda x: f"users:{x}")
def behacom_dataset(request):
    """Hourly user laptop behavior dataset.

    Train-test split by `session_id` (`user`, `date-hour`).
    Multi-class classification (softmax). Classify each test session into one of the 11 users.

    11 users, ~12,000 dimensions (e.g. RAM usage, CPU usage, mouse location),
    ~5000 timestamps. Relevant to IoT and productivity.

    We take samples of top 3, 6, 9 users by days observed.

    Note: we drop user 2 as they only have 1 day observed.
    """

    entity_col = "session_id"
    time_col = "timestamp"
    label_col = "user"
    n_users = request.param
    start = 0
    step = 1

    dataset = request.config.cache.get("dataset/behacom", None)
    if dataset is not None:
        logging.info("🗂️ Loading %r dataset from cache", "behacom")
        X_train, X_test, y_train, y_test = decode_dataset(dataset)
        logging.info("✅ %r dataset loaded", "behacom")
    else:
        top_k_users = (
            pl.scan_parquet("data/behacom.parquet")
            .select(["user", "timestamp"])
            .groupby("user")
            .agg((pl.col("timestamp").max() - pl.col("timestamp").min()).dt.days().alias("days"))
            .top_k(k=n_users, by="days")
            .collect(streaming=True)
        )
        logging.info("🎲 Selected users: %s", top_k_users.to_dicts())
        data = (
            pl.scan_parquet("data/behacom.parquet")
            .filter(pl.col("user").is_in(top_k_users.get_column("user")))
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
            # Create session id
            .with_columns(date=pl.col("timestamp").dt.strftime("%Y%m%d").cast(pl.Categorical).cast(pl.Int32))
            .with_columns(
                session_id=pl.struct([label_col, "date"]).hash().cast(pl.Utf8).cast(pl.Categorical).to_physical(),
                # Convert timestamp into hours
                timestamp=pl.col("timestamp").dt.hour()
            )
            # Defensive sort
            .sort([entity_col, time_col])
            .set_sorted([entity_col, time_col])
            .select([
                entity_col,
                time_col,
                ps.numeric().exclude([entity_col, time_col])
            ])
            .collect(streaming=True)
        )
        X_train, X_test, y_train, y_test = split_autocorrelated_data(data, label_cols=[label_col])
        # Cache dataset
        cached_dataset = encode_dataset(X_train, X_test, y_train, y_test)
        request.config.cache.set("dataset/behacom", cached_dataset)

    preview_dataset("behacom", X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test, start, step


@pytest.fixture(params=[0.2, 0.4, 0.8, 1.0], ids=lambda x: f"fraction:{x}")
def elearn_dataset(request):
    """Massive e-learning exam score prediction from Kaggle.

    Multi-output classification problem. Predict 18 questions win / loss across sessions.
    Dataset contains 23,562 sessions intotal.

    We take a random sample of 0.2, 0.4, 0.8, and 1.0 sessions.

    Relevant to education, IoT, and online machine learning.
    Link: https://www.kaggle.com/competitions/predict-student-performance-from-game-play/data
    """
    # Game Walkthrough:
    # https://www.kaggle.com/competitions/predict-student-performance-from-game-play/discussion/384796
    entity_col = "session_id"
    time_col = "index"
    sample_fraction = request.param

    dataset = request.config.cache.get("dataset/elearn", None)
    if dataset is not None:
        logging.info("🗂️ Loading %r dataset from cache", "elearn")
        X_train, X_test, y_train, y_test = decode_dataset(dataset)
        logging.info("✅ %r dataset loaded", "elearn")
    else:
        # Pivot from long to wide format (for multioutput classification)
        labels = (
            pl.read_parquet("data/elearn_labels.parquet")
            .pivot(index="session_id", columns="question_id", values="correct")
            .select(["session_id", pl.all().exclude("session_id").prefix("question_")])
        )
        sampled_session_ids = labels.get_column("session_id").unique().sample(fraction=sample_fraction)
        logging.info("🎲 Selected %s / %s sessions (%.2f)", len(sampled_session_ids), len(labels), sample_fraction)
        data = (
            pl.read_parquet("data/elearn.parquet")
            # Sample session IDs
            .filter(pl.col(entity_col).is_in(sampled_session_ids))
            # Drop columns if all null
            .pipe(lambda df: df.select([s for s in df if s.null_count() < s.len()]))
            .lazy()
            # Select numeric only
            .select([entity_col, time_col, ps.numeric()])
            # Presort to speed up join
            .sort([entity_col, time_col])
            .set_sorted([entity_col, time_col])
            # Join with multioutput labels
            .join(labels.lazy(), how="left", on="session_id")
            # Forward fill observations
            .fill_null(strategy="forward")
            # BUG: We groupby max to remove duplciated observations
            .groupby([entity_col, time_col])
            .agg(pl.all().max())
            .collect(streaming=True)
        )
        X_train, X_test, y_train, y_test = split_iid_data(data, label_cols=labels.columns[1:])
        # Cache dataset
        cached_dataset = encode_dataset(X_train, X_test, y_train, y_test)
        request.config.cache.set("dataset/elearn", cached_dataset)

    start, step = step_from_windows(X_train, min_windows=24, start_fraction=0.5)
    preview_dataset("elearn", X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test, start, step


def _fit_predict_score(
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    start: int,
    step: int,
) -> pl.DataFrame:

    # Set expanding window configs
    entity_col, time_col = y_test.columns[:2]
    # Fit
    model = Hindsight(start=start, step=step, random_state=42)
    model.fit(X=X_train, y=y_train)

    # Predict-score
    logging.info("💯 Scoring Hindsight...")
    metrics = [accuracy_score, balanced_accuracy_score]
    # NOTE: Setting keep_true=True caches y_pred in model
    full_scores, y_pred = model.score(X=X_test, y=y_test, keep_pred=True, metrics=metrics)
    session_scores = model.score(X=X_test, y=y_test, by=entity_col, metrics=metrics)
    period_scores = model.score(X=X_test, y=y_test, by=[entity_col, time_col], metrics=metrics)

    logging.info("📹 Predicted labels preview:\n%s", y_pred)

    return y_pred, full_scores, session_scores, period_scores


def test_hindsight_on_behacom(behacom_dataset):
    X_train, X_test, y_train, y_test, start, step = behacom_dataset
    y_pred, full_scores, session_scores, period_scores = _fit_predict_score(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        start=start,
        step=step,
    )


def test_hindsight_on_elearn(elearn_dataset):
    X_train, X_test, y_train, y_test, start, step = elearn_dataset
    y_pred, full_scores, session_scores, period_scores = _fit_predict_score(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        start=start,
        step=step,
    )
