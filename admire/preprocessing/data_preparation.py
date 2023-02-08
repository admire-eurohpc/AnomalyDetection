from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import List


def prep_data(filename : str, col_list : List[str]) -> [np.array, pd.DataFrame]:
    df = pd.read_csv(filename)
    df = df[col_list]

    le = preprocessing.LabelEncoder()
    scaler = StandardScaler()

    for col in ['flags', 'group', 'partition']:
        df[col] = le.fit_transform(df[col])

    df_pca = df.to_numpy().astype(np.int64)
    df_pca = scaler.fit_transform(df_pca)
    return df_pca, df