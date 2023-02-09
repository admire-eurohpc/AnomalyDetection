import sys
import os
import pandas as pd

from admire.models.data_exploration import _pca, _kmeans
from admire.preprocessing.data_preparation import NumFiltering, DataPreparation
from admire.visualisation.visualisation import visualisation #TODO change names to be more informative and function to be more general

sys.path.append('../../')

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

print(ROOT_DIR)

def clustering():
    df_raw = pd.read_csv(os.path.join(ROOT_DIR, 'data', 'jobs_flattened_cleaned.csv'))

    energy_filter = NumFiltering(df_raw, 'steps-tres-consumed-max-energy')
    df_raw = energy_filter.filter_values('down', [5000])
    time_filter = NumFiltering(df_raw, 'steps-time-total-seconds')
    df_raw = time_filter.filter_values('up', [600])

    data = DataPreparation(df_raw,
                           ['allocation_nodes', 'flags', 'group', 'partition', 'priority', 'required-CPUs',
                            'steps-time-elapsed', 'steps-time-system-seconds', 'steps-time-user-seconds',
                            'steps-tres-requested-max-cpu', 'steps-tres-requested-max-mem',
                            'steps-tres-requested-max-fs', 'steps-tres-consumed-max-energy',
                            'steps-tres-consumed-max-fs']).prepare(encoding='label_encoding')

    var, pca = _pca(data)
    clusters, labels, centroids = _kmeans(pca)
    print("PCA explains: ", var, 'variance.')
    visualisation(clusters, labels)


if __name__ == "__main__":
    clustering()