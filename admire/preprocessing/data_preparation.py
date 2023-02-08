from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import List


 #TODO Napisac funkcje odfiltrowujaca wysokie pomiary steps-tres-consumed-max-energy
 #TODO Odfiltrowac kilkusekundowe joby
def prep_data(filename : str, col_list : List[str]) -> [np.array, pd.DataFrame]:
    df_raw = pd.read_csv(filename)
    df = df_raw[col_list]

    le = preprocessing.LabelEncoder()
    scaler = StandardScaler()

    for col in ['flags', 'group', 'partition']:
        if col in col_list:
            df[col] = le.fit_transform(df[col])

    df_pca = df.to_numpy().astype(np.int64)
    df_pca = scaler.fit_transform(df_pca)
    return df_pca, df_raw

def extract_nodes_from_node(node_string: str) -> list:
    return_nodes = list()

    if '[' in node_string:
        comma_split = node_string[2:len(node_string) - 1].split(',')
        for node in comma_split:
            if '-' in node:
                range_low, range_up = node.split('-')
                for i in range(int(range_low), int(range_up) + 1, 1):
                    return_nodes.append(f'e{i}')
            else:
                return_nodes.append(f'e{node}')
    else:
        return_nodes.append(node_string)

    return return_nodes

def create_node_list(nodes: pd.Series) -> np.array:
    all_count = list()
    split_nodes = list()

    for node_String in nodes.to_list():
        nodes = extract_nodes_from_node(node_String)
        all_count += nodes
        split_nodes.append(nodes)

    print(type(split_nodes))

    node_series = np.asarray(split_nodes)
    return node_series


