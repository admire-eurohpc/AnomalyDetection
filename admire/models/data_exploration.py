import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def _pca(df: pd.DataFrame) -> [np.array, np.array]:
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(df)
    return pca.explained_variance_ratio_, X_pca

def _kmeans(arr: np.array) -> [np.array, np.array, np.array]:
    kmeans = KMeans(n_clusters=8, random_state=0, n_init=20)
    out = kmeans.fit_transform(arr)
    label = kmeans.fit_predict(out)
    centroids = kmeans.cluster_centers_
    return out, label, centroids