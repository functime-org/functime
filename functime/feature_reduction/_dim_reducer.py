import polars as pl

class DimensionReducer:
    def __init__(self):
        self.state_model = None

    def fit_pca(self, X: pl.DataFrame, dim: int, **kwargs):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim)
        pca.fit(X)
        self.state_model = pca
        

    # Other dimension reduction related methods...
    def fit_t_sne(self, features, method='PCA', **kwargs):
        pass
