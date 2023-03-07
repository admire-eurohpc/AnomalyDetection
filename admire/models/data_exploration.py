import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


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
