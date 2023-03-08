import sys
import os
import pandas as pd
import datetime
import numpy as np

from tslearn.clustering import TimeSeriesKMeans
from collections import Counter
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from admire.models.data_exploration import _pca, _kmeans
from admire.preprocessing.data_preparation import NumFiltering, DataPreparation
from admire.visualisation.visualisation import visualisation #TODO change names to be more informative and function to be more general

sys.path.append('../../')

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
ENCODING = 'label_encoding'
print(ROOT_DIR)

def clustering():
    df_trimmed = pd.read_parquet(os.path.join(ROOT_DIR, 'data', '1-22.02.2023_tempdata_trimmed.parquet'))
    df_trimmed['date'] = pd.to_datetime(df_trimmed['date'])
    x = []
    print("data loaded and prepared")
    df_trimmed = df_trimmed[df_trimmed['date'].dt.date == datetime.date(2023, 2, 5)]
    for elem in df_trimmed['hostname'].unique():
        df_temp = df_trimmed[df_trimmed['hostname'] == elem]
        if len(df_temp['power']) == 144:
            res = df_temp['power'].reset_index(drop=True)
            res.name=elem
            x.append(res)
    model = TimeSeriesKMeans(n_clusters=8, metric="dtw", max_iter=10)
    x_ = np.reshape(x, (np.shape(x)[0], np.shape(x)[1], 1))
    model.fit(x_)



    fig = make_subplots(rows=4, cols=2)
    for i in range(0,8):
        for j, label in enumerate(model.labels_):
            if i == label:
                fig.append_trace(go.Scatter(y = x[j], mode="lines"), row=i//2+1, col=i%2+1)
    fig.show()

if __name__ == "__main__":
    clustering()