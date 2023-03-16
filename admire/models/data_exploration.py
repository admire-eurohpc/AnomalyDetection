import numpy as np
import pandas as pd
import plotly.figure_factory as ff
from typing import Any

from tslearn.metrics.dtw_variants import cdist_dtw
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import LocalOutlierFactor
from scipy.cluster.hierarchy import single, complete, average, ward, fcluster

from collections import Counter



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


def _logistic_regression(x_train: np.array, y_train: np.array, x_test: np.array, y_test:np.array, encoding: str):
    if encoding == 'label_encoding' or encoding == 'None':
        clf = LogisticRegression(random_state=0, verbose=10, n_jobs=-1).fit(x_train, y_train)
        print(clf.score(x_test, y_test))
    elif encoding == "onehot_encoding":
        raise ValueError('Onehot encoding not supported')
        # clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(x_train, y_train)
        # print(clf.score(x_test, y_test))

def time_series_clustering(data: np.array, clustering: str, n_clusters: int = 8) -> Any:
    if clustering == 'kmeans':
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10)
    elif clustering == "dbscan":
        norm = Normalizer()
        data = cdist_dtw(data, n_jobs=-1, verbose=1)
        data = norm.fit_transform(data)
        model = DBSCAN(eps=0.0017, min_samples=8, metric="precomputed").fit(data)
        labels = model.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(n_clusters)        
    model.fit(data)
    return model, n_clusters

def hierarchical_clustering(data: np.array, method: str='complete'):
    distance_matrix = cdist_dtw(data, n_jobs=-1, verbose=1)
    if method == 'complete':
        Z = complete(distance_matrix)
    if method == 'single':
        Z = single(distance_matrix)
    if method == 'average':
        Z = average(distance_matrix)
    if method == 'ward':
        Z = ward(distance_matrix)
    
    fig = ff.create_dendrogram(Z)
    fig.show()
    n_clusters = 7
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')

    return Z, cluster_labels, n_clusters

def local_outlier_factor(data: np.array) -> Any:
    data = cdist_dtw(data, n_jobs=-1, verbose=0)
    model = LocalOutlierFactor(n_neighbors=40, contamination=0.05, metric='precomputed')
    labels = model.fit_predict(data)
    return model, labels, len(set(labels))
