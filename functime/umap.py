from typing import Optional

import numpy as np


def auto_umap(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_dims: int = 3,
    n_neighbors: int = 3,
    min_dist: float = 0.1,
    random_state: Optional[int] = None,
) -> np.ndarray:
    import umap

    model = umap.UMAP(
        n_components=n_dims,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="manhattan",
        random_state=random_state,
    )
    embs = model.fit_transform(X=X, y=y)
    return embs
