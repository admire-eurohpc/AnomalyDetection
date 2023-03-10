import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from typing import Any


def visualisation(arr: np.array, labels: Any):
    print(type(labels))
    title = 'Kmeans on PCA'
    fig = px.scatter_3d(arr, x=0, y=1, z=2, color=labels, title=title)
    fig.update_traces(marker=dict(size=2))
    fig.show()

def sublots_clustering(n_clusters: int, model_labels: Any, data: np.array):
    '''
    n_clusters : how many cluster figures will plotly generate
    model_labels : labels for different clusters in data
    x : data
    '''

    fig = make_subplots(rows=n_clusters//2+1, cols=2)
    for i, elem in enumerate(set(model_labels)):
        for j, label in enumerate(model_labels):
            if elem == label:
                fig.append_trace(go.Scatter(y = data[j], mode="lines"), row=i//2+1, col=i%2+1)
    fig.show()