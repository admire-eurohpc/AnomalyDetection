import sys
sys.path.append('../../')
import os
from admire.models.data_exploration import _pca, _kmeans
from admire.preprocessing.data_preparation import prep_data
from admire.visualisation.visualisation import visualisation #TODO change names to be more informative and function to be more general

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

print(ROOT_DIR)

def clustering():
    df = prep_data(os.path.join(ROOT_DIR, 'data', 'jobs_cleaned_redundant.csv'),
                   ['allocation_nodes', 'flags', 'group', 'partition', 'priority', 'required-CPUs', 'steps-time-elapsed'
                    , 'steps-time-system-seconds', 'steps-time-user-seconds', 'steps-tres-requested-max-cpu',
                    'steps-tres-requested-max-mem', 'steps-tres-requested-max-fs', 'steps-tres-consumed-max-energy',
                    'steps-tres-consumed-max-fs'])
    var, pca = _pca(df)
    clusters, labels, centroids = _kmeans(pca)
    print("PCA explains: ", var, 'variance.')
    visualisation(clusters, labels)


if __name__ == "__main__":
    clustering()