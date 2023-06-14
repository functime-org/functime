import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from functime.embeddings import embed

DATA_URL = "https://github.com/indexhub-ai/functime/raw/main/data/"


@pytest.fixture(scope="module")
def gunpoint_dataset():
    """Equal length, univariate time series."""
    X_y_train = pl.read_parquet(f"{DATA_URL}/gunpoint_train.parquet")
    X_y_test = pl.read_parquet(f"{DATA_URL}/gunpoint_test.parquet")
    X_train = X_y_train.select(pl.all().exclude("label"))
    y_train = X_y_train.select("label")
    X_test = X_y_test.select(pl.all().exclude("label"))
    y_test = X_y_test.select("label")
    return X_train, X_test, y_train, y_test


def test_minirocket_on_gunpoint(gunpoint_dataset):
    # each row is 1 time series, each column is 1 time point
    X_training, X_test, Y_training, Y_test = gunpoint_dataset

    # Keep everything as np.ndarray
    X_training = X_training.to_numpy()
    X_test = X_test.to_numpy()
    Y_training = Y_training.to_numpy().ravel()
    Y_test = Y_test.to_numpy().ravel()

    # Minirocket takes in numpy array with columnar format
    X_training_transform = embed(X_training, model="minirocket")

    # test shape of transformed training data -> (number of training
    # examples, nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_training_transform.shape, (len(X_training), 9_996))

    # fit classifier
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )
    classifier.fit(X_training_transform, Y_training)

    # transform test data
    X_test_transform = embed(X_test, model="minirocket")

    # test shape of transformed test data -> (number of test examples,
    # nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_test_transform.shape, (len(X_test), 9_996))

    # predict (alternatively: 'classifier.score(X_test_transform, Y_test)')
    predictions = classifier.predict(X_test_transform)
    accuracy = accuracy_score(predictions, Y_test)

    # test predictions (on Gunpoint, should be > 99% accurate)
    assert accuracy > 0.97


@pytest.mark.parametrize("size", [25, 50, 75, 100, 125, 150])
@pytest.mark.benchmark(group="embeddings")
def test_mr(benchmark, gunpoint_dataset, size):
    # each row is 1 time series, each column is 1 time point
    X_training, _, _, _ = gunpoint_dataset

    # Keep everything as np.ndarray
    X_training = X_training.to_numpy()[:size]

    # Minirocket takes in numpy array with columnar format
    X_training_transform = benchmark(embed, X_training)

    # test shape of transformed training data -> (number of training
    # examples, nearest multiple of 84 < 10,000)
    np.testing.assert_equal(X_training_transform.shape, (len(X_training), 9_996))
