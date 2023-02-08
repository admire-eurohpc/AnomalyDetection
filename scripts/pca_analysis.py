import sys
import os
sys.path.append('../../')
from admire.models.data_exploration import _pca
from admire.preprocessing.data_preparation import prep_data #TODO change function_name to be more informative, merge with the rest of preprocessing code from Ignacy
from admire.visualisation.visualisation import visualisation #TODO change names to be more informative and function to be more general

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))



def pca_analysis():
    data, df_raw = prep_data(os.path.join(ROOT_DIR, 'data', 'jobs_cleaned_redundant.csv'),
                   ['allocation_nodes', 'flags', 'group', 'partition', 'priority', 'required-CPUs', 'steps-time-elapsed'
                    , 'steps-time-system-seconds', 'steps-time-user-seconds', 'steps-tres-requested-max-cpu',
                    'steps-tres-requested-max-mem', 'steps-tres-requested-max-fs', 'steps-tres-consumed-max-energy',
                    'steps-tres-consumed-max-fs'])
    var, pca = _pca(data)
    print("PCA explains: ", var, 'variance.', type(pca))

    visualisation(pca, df_raw['steps-time-system-seconds'])


if __name__ == "__main__":
    pca_analysis()
