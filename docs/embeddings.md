# Time Series Embeddings

## What are embeddings?

Time-series embeddings measure the relatedness of time-series.
Embeddings are more accurate and efficient compared to statistical methods (e.g. Catch22) for characteristing time-series.[^1]
Embeddings have applications across many domains from finance to IoT monitoring.
They are commonly used for the following tasks:

- **Search:** Where time-series are ranked by similarity to a given time-series
- **Classification:** Where time-series are grouped together by matching patterns
- **Clustering:** Where time-series are assigned labels (e.g. normal vs irregular heart rate)
- **Anomaly detection:** Where outliers with unexpected regime / trend changes are identified

!!! example "To see time-series embeddings in action, check out our code examples"

    - **Search** with e-commerce data
    - **Classification** with health biometrics
    - **Clustering** with S&P 500 stock prices
    - **Anomaly Detection** with server loads and IoT data

    [Browse use-cases](#what-are-the-use-cases){ .md-button .md-button--primary }


## How to compute embeddings?

## How are embeddings computed?

`functime` offers `RustyRocket`, which is currently the fastest implementation of MINIROCKET[^1] (MINImally RandOm Convolutional KErnel Transform). The MINIROCKET algorithm consistently tops time-series classification benchmarks in speed and accuracy.

[^1]: Dempster, A., Schmidt, D. F., & Webb, G. I. (2021, August). Minirocket: A very fast (almost) deterministic transform for time series classification. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 248-257).


## What are the use-cases?

### Search

### Classification

### Clustering

### Anomaly Detection

## How can I retrieve K-nearest embeddings quickly?

To search over many embeddings quickly, we recommend using a vector database.
Our current recommendation is [LanceDB](https://github.com/lancedb/lancedb) for its first-class support for time-travel, fast distance metrics, and easy-to-use API.
