from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import List
import abc


 #TODO Napisac funkcje odfiltrowujaca wysokie pomiary steps-tres-consumed-max-energy
 #TODO Odfiltrowac kilkusekundowe joby

class Filtering:
    def __init__(self, df: pd.DataFrame, col_name : str):
        self.df = df
        self.col_name = col_name

    @abc.abstractmethod
    def filter_values(self):
        pass

class NumFiltering(Filtering):
    def __init__(self, thresh_type: str, thresh: List, df: pd.DataFrame, col_name: str):
        self.thresh = thresh
        self.thresh_type = thresh_type
        super(NumFiltering, self).__init__(df, col_name)


    def filter_values(self):
        if len(self.thresh) == 0:
            raise ValueError('Provide a list with at least one filter value')
        elif len(self.thresh) == 1:
            if self.thresh_type == 'up':
                self.df = self.df.loc[self.df[self.col_name] > self.thresh[0]]
            elif self.thresh_type == 'down':
                self.df = self.df.loc[self.df[self.col_name] < self.thresh[0]]
            else:
                raise ValueError('Threshold type takes only "up" or "down" value.')
        elif len(self.thresh) == 2:
            if self.thresh[0] < self.thresh[1]:
                self.df = self.df.loc[(self.df[self.col_name] > self.thresh[0]) &
                                      (self.df[self.col_name] > self.thresh[1])]
            else:
                raise ValueError('First threshold value should be lower than second.')
        else:
            raise ValueError('Provided too many filter values. Provide a list with 1 or 2 filter values')


def prep_data(df: pd.DataFrame, col_list : List[str]) -> [np.array, pd.DataFrame]:
    df = df[col_list]

    le = preprocessing.LabelEncoder()
    scaler = StandardScaler()

    for col in ['flags', 'group', 'partition']:
        if col in col_list:
            df[col] = le.fit_transform(df[col])

    df_pca = df.to_numpy().astype(np.int64)
    df_pca = scaler.fit_transform(df_pca)
    return df_pca

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


