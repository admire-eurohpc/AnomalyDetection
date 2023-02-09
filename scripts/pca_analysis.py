import sys
import os
import pandas as pd

from admire.models.data_exploration import _pca
from admire.preprocessing.data_preparation import DataPreparation #TODO change function_name to be more informative, merge with the rest of preprocessing code from Ignacy
from admire.visualisation.visualisation import visualisation #TODO change names to be more informative and function to be more general

sys.path.append('../../')

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))



def pca_analysis():
    df_raw = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'jobs_flattened_cleaned.csv'))
    data = DataPreparation(df_raw,
                           ['allocation_nodes', 'flags', 'group', 'partition', 'priority', 'required-CPUs',
                            'steps-time-elapsed', 'steps-time-system-seconds', 'steps-time-user-seconds',
                            'steps-tres-requested-max-cpu', 'steps-tres-requested-max-mem',
                            'steps-tres-requested-max-fs', 'steps-tres-consumed-max-energy',
                            'steps-tres-consumed-max-fs']).prepare(encoding='label_encoding')
    var, pca = _pca(data)
    print("PCA explains: ", var, 'variance.', type(pca))

    visualisation(pca, df_raw['partition'])


if __name__ == "__main__":
    pca_analysis()
