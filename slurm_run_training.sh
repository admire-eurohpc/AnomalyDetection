#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH --job-name=admire
#SBATCH --gres gpu:1
#SBATCH -p tesla
#SBATCH --output=slurm_logs/output-%A.txt
#SBATCH --error=slurm_logs/errors/error-%A.txt


echo "Activating the virtual environment"
source /home/users/ignacys/pl0134-01/project_data/pyvenv/bin/activate
echo "Virtual environment activated"

# Prepare data
echo "Preparing data"
python admire/preprocessing/ae_data_prep.py
echo "Finished preparing data"

# First run CNN version
echo "Running training CNN version"
python admire/models/ae_train.py --model_type CNN
echo "Finished running training CNN version"


# Second run LSTMCNN version
echo "Running training LSTMCNN version"
python admire/models/ae_train.py --model_type LSTMCNN
echo "Finished running training LSTMCNN version"

# Third run LSTMPLAIN version
echo "Running training LSTMPLAIN version"
python admire/models/ae_train.py --model_type LSTMPLAIN
echo "Finished running training LSTMPLAIN version"

# Print when finished
echo "-----------------------------"
echo "Finished running training on the remote server"