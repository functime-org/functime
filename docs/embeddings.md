# Time Series Embeddings

## What are embeddings?

Time-series embeddings measure the relatedness of time-series.
Embeddings are **orders of magnitude** more accurate and efficient compared to statistical methods (e.g. Catch22) for characterizing time-series.[^1]
Embeddings have applications across many domains from finance to IoT monitoring.
They can be used to solve the following predictive tasks:

- **Matching:** Where time-series are ranked by similarity to a given time-series
- **Win / Loss:** Where a binary outcome (e.g. sports game) is predicted using event histories (e.g. trajectory of the ball)
- **Classification:** Where time-series are assigned labels (e.g. normal vs irregular heart rate)
- **Clustering:** Where time-series are grouped together by matching patterns
- **Anomaly detection:** Where outliers with unexpected trend changes are identified

!!! tip "To see time-series embeddings in action, check out our code examples"

    - **Classification** with health biometrics (fetal heartbeat data)
    - **Clustering** with S&P 500 stock prices
    - **Anomaly detection** with user behavior on their laptops

    Other potential use-cases for time series embeddings include:
    - Churn prevention by matching purchasing patterns of active users to past users that churned
    - Classifying measurements over time from IoT / robotic sensors to different model types or environments

    [Browse use-cases](#what-are-the-use-cases){ .md-button .md-button--primary }

## How to compute embeddings?

The `functime.embeddings.embed()` function takes a **wide dataset** where each row represents a single time-series.

!!! example "Wide data example"
    The following dataset represents velocity measurements from two robots (label 1 and label 2) over 150 time periods (columns t0, t1, ..., t149) and 75 trials (rows).
    ```
    >>> X_y_wide = pl.read_parquet("https://bit.ly/gunpoint-train")
    >>> X_y_wide
    shape: (150, 151)

    label     t0        t1     ...    t148      t149
    --------------------------------------------------
    1     -1.125013 -1.131338  ... -1.206178 -1.218422
    2     -0.626956 -0.625919  ... -0.612058 -0.606422
    2     -2.001163 -1.999575  ... -1.071147 -1.323383
    1     -1.004587 -0.999843  ... -1.044226 -1.043262
    1     -0.742625 -0.743770  ... -0.670519 -0.657403
    ...         ...       ...  ...       ...       ...
    2     -0.580006 -0.583332  ... -0.548831 -0.553552
    1     -0.728153 -0.730242  ... -0.686448 -0.690183
    2     -0.738012 -0.736301  ... -0.608616 -0.612177
    2     -1.265111 -1.256093  ... -1.193374 -1.192835
    1     -1.427205 -1.408303  ... -1.153119 -1.222043
    ```

```python
import functime

X = X_y_wide.select(pl.all().exclude("label"))
X_embs = functime.embeddings.embed(X, model="minirocket")
```

The embeddings can be reduced into 2D / 3D and visualized with a scatter plot.
![Embeddings](img/embeddings_clip.gif)

## How are embeddings computed?

`functime` offers `RustyRocket`, which is currently the fastest implementation of MINIROCKET[^1] (MINImally RandOm Convolutional KErnel Transform). The MINIROCKET algorithm consistently tops time-series classification benchmarks in both speed and accuracy.

[^1]: Dempster, A., Schmidt, D. F., & Webb, G. I. (2021, August). Minirocket: A very fast (almost) deterministic transform for time series classification. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 248-257).

## What are the use-cases?

### Classification (Health)

In this example, we classify 750 fetal electrocardiogram visits (fetal heartbeat measurements) from 42 women.

```python
import polars as pl
import functime
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# Load dataset (150 observations, 150 timestamps)
X_y_train = pl.read_parquet("https://github.com/descendant-ai/functime/raw/main/data/fetal_test.parquet")
X_y_test = pl.read_parquet("https://github.com/descendant-ai/functime/raw/main/data/fetal_train.parquet")

# Train-test split
X_train, y_train = (
    X_y_train.select(pl.all().exclude("label")),
    X_y_train.select("label")
)
X_test, y_test = (
    X_y_test.select(pl.all().exclude("label")),
    X_y_test.select("label")
)

X_train_embs = functime.embeddings.embed(X_train, model="minirocket")

# Fit classifier on the embeddings
classifier = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
)
classifier.fit(X_train_embs, y_train)

# Predict and
X_test_embs = embed(X_test, model="minirocket")
labels = classifier.predict(X_test_embs)
accuracy = accuracy_score(predictions, y_test)
```

### Clustering (Finance)

In this example, we cluster S&P 500 companies into groups with similar price patterns.

```python
import functime
import polars as pl
from hdbscan import HDBSCAN
from umap import UMAP
from functime.preprocessing import roll

# Load S&P500 panel data from 2022-06-01 to 2023-06-01
# Columns: ticker, time, price
y = pl.read_parquet("https://github.com/descendant-ai/functime/raw/main/data/sp500.parquet")

# Reduce noise by smoothing the time series using
# functime's `roll` function: 60-days moving average
y_ma_60 = (
    y.pipe(roll(window_sizes=[60], stats=["mean"], freq="1d"))
    .drop_nulls()
    # Pivot from panel to wide format
    .pivot(
        values="price__rolling_mean_60",
        columns="time",
        index="ticker"
    )
    # Remember all functime transforms are lazy!
    .collect()
)

# Create embeddings
embeddings = functime.embeddings.embed(y_ma_60, model="minirocket")

# Reduce dimensionality with UMAP
reducer = UMAP(n_components=500, n_neighbors=10, metric="manhattan")
umap_embeddings = reducer.fit_transform(embeddings)

# Cluster with HDBSCAN
clusterer = HDBSCAN(metric="minkowski", p=1)
estimator.fit(X)

# Get predicted cluster labels
labels = estimator.predict(X)
```

### Anomaly Detection (User Behavior)

In this example, we compare time series embeddings for laptop activity (CPU and RAM usage) across 12 users.
Anomalies are identified by unusual distance away from the midpoint of all embeddings.
In particular, we use the 1.5 IQR (interquartile range) method given the distribution of distances from the midpoint.

```python
import functime
import polars as pl
import numpy as np
from scipy.stats import iqr

# Load memory usage data
y = pl.read_parquet("https://github.com/descendant-ai/functime/raw/main/data/laptop.parquet", columns=["user", "timestamp", "memory"])

# Create embeddings
embeddings = functime.embeddings.embed(y, model="minirocket")

# Compute midpoint and distances from midpoint
midpoint = np.mean(embeddings, axis=0)
distances = np.linalg.norm(embeddings-midpoint, axis=1)

# Compute IQR
q_low = np.quantile(distances, q=0.1)
q_high = np.quantile(distances, q=0.9)

# Identify outliers
outliers = (
    pl.DataFrame({"user": range(12), "distance": distances})
    .filter(pl.col("distance") < q_high & pl.col("distance") > q_high)
)
```

## What's next?

Time-series embeddings are a disruptive new technique for data mining with extremely large numbers of time-series.
If you have an interesting use-case, we would love to hear from you!
Let's chat over a [15 minute call](https://calendly.com/functime-indexhub).

## How can I retrieve K-nearest embeddings quickly?

To search over many embeddings quickly, we recommend using a vector database.
Our current recommendation is [LanceDB](https://github.com/lancedb/lancedb) for its first-class support for time-travel, fast distance metrics, and easy-to-use API.
