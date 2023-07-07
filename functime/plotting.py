import numpy as np
import plotly.express as px


def plot_scatter(X: np.ndarray, y: np.ndarray):
    fig = px.scatter_3d(
        x=X[:, 0],
        y=X[:, 1],
        z=X[:, 2],
        color=y.squeeze().astype(str),  # Discrete colors
        opacity=0.5
    )
    fig.update_traces(marker_size=3)
    return fig
