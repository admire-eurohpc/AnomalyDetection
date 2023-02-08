import numpy as np
import plotly.express as px
from typing import Any


def visualisation(arr: np.array, labels: Any):
    print(type(labels))
    title = 'Kmeans on PCA'
    fig = px.scatter_3d(arr, x=0, y=1, z=2, color=labels, title=title)
    fig.update_traces(marker=dict(size=2))
    fig.show()