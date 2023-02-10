from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
from typing import List
import abc
pd.options.mode.chained_assignment = None

 #TODO Napisac funkcje odfiltrowujaca wysokie pomiary steps-tres-consumed-max-energy
 #TODO Odfiltrowac kilkusekundowe joby

class Filtering:
    def __init__(self, df: pd.DataFrame, col_name : str):
        self.df = df
        self.col_name = col_name

    @abc.abstractmethod
    def filter_values(self, thresh_type: str, thresh: List,):
        pass


class NumFiltering(Filtering):
    def filter_values(self, thresh_type: str, thresh: List):
        if len(thresh) == 0:
            raise ValueError('Provide a list with at least one filter value')
        elif len(thresh) == 1:
            if thresh_type == 'up':
                self.df = self.df.loc[self.df[self.col_name] > thresh[0]]
                return self.df
            elif thresh_type == 'down':
                self.df = self.df.loc[self.df[self.col_name] < thresh[0]]
                return self.df
            else:
                raise ValueError('Threshold type takes only "up" or "down" value.')
        elif len(thresh) == 2: #Is this functionality necessary? We can bring all operations to multiple usage of single filters. TBD
            if thresh[0] < thresh[1]:
                self.df = self.df.loc[(self.df[self.col_name] > thresh[0]) &
                                      (self.df[self.col_name] < thresh[1])]
                return self.df
            else:
                raise ValueError('First threshold value should be lower than second.')
        else:
            raise ValueError('Provided too many filter values. Provide a list with 1 or 2 filter values')


class DataPreparation:
    def __init__(self, df: pd.DataFrame, col_list: List[str]):
        self.df = df
        self.col_list = col_list

    @staticmethod
    def choose_columns(df: pd.DataFrame, col_list: List[str]) -> pd.DataFrame:
        df = df[col_list]
        return df

    @staticmethod
    def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = []
        df['flags'] = df['flags'].apply(lambda x: x[22:41])
        for column in df.columns:
            if column.startswith('time') or column.startswith('wckey'):  # time is redundant, more information is held in steps-time, wckey holds no info
                drop_cols.append(column)
            if len(df[column].unique()) < 2:  # dropping cols with only one value ([], {}, 0, nan)
                drop_cols.append(column)

        df = df.drop(columns=drop_cols)
        return df

    @staticmethod
    def label_encoding(df: pd.DataFrame):
        le = LabelEncoder()

        for col in df.columns:
            try:
                df[col].astype(int)
            except ValueError:
                if type(df[col].iloc[0]) is str:
                    df[col] = le.fit_transform(df[col])
                else:
                    print(f'Wrong column type.')

        df = df.to_numpy().astype(np.int64)
        return df

    @staticmethod
    def onehot_encoding(df: pd.DataFrame): #TODO fix onehot_encoding
        enc = OneHotEncoder()

        for col in df.columns:
            try:
                df[col].astype(int)
            except ValueError:
                if type(df[col].iloc[0]) is str:
                    df[col] = enc.fit_transform(df[col])
                else:
                    print(f'Wrong column type.')

        df = df.to_numpy().astype(np.int64)
        return df

    @staticmethod
    def standard_scaling(arr: np.array):
        scaler = StandardScaler()
        arr = scaler.fit_transform(arr)
        return arr

    def prepare(self, encoding):
        df = self.choose_columns(self.df, self.col_list)
        df = self.clean_dataset(df)
        if encoding == 'label_encoding':
            arr = self.label_encoding(df)
        elif encoding == 'onehot_encoding':
            arr = self.onehot_encoding(df)
        else:
            raise ValueError('Encoding accepts only "label_encoding" or "onehot_encoding" values')
        arr = self.standard_scaling(arr)
        return arr

class MLDataPreparation(DataPreparation): #TODO finish MLDataPreparation
    def __init__(self, df: pd.DataFrame, col_list: List[str]):
        super(MLDataPreparation, self).__init__(df, col_list)


def prep_data(df: pd.DataFrame, col_list : List[str]) -> [np.array, pd.DataFrame]: #TODO fix warnings
    df = df[col_list]

    le = preprocessing.LabelEncoder()
    scaler = StandardScaler()

    for col in df.columns:
        try:
            df[col].astype(int)
        except ValueError:
            if type(df[col].iloc[0]) is str:
                df[col] = le.fit_transform(df[col])
            else:
                print(f'Wrong column type.')


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


