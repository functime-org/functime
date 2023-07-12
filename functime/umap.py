import numpy as np
import umap


def auto_umap(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = 3,
    min_dist: float = 0.1,
    n_dims: int = 3,
) -> np.ndarray:
    model = umap.UMAP(
        n_components=n_dims,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="manhattan",
    )
    embs = model.fit_transform(X=X, y=y)
    return embs
