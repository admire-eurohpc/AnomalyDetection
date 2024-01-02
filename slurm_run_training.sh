#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH --job-name=admire
#SBATCH --gres gpu:1
#SBATCH -p tesla

echo "Activating the virtual environment"
source /home/users/ignacys/pl0134-01/project_data/pyvenv/bin/activate
echo "Virtual environment activated"

# Prepare data
echo "Preparing data"
python admire/preprocessing/ae_data_prep.py
echo "Finished preparing data"

# First run CNN version
echo "Running training CNN version"
cp config_cnn.ini config.ini
python admire/models/ae_train.py --model_type CNN
echo "Finished running training CNN version"


# Second run LSTMCNN version
echo "Running training LSTMCNN version"
cp config_lstm.ini config.ini
python admire/models/ae_train.py --model_type LSTMCNN
echo "Finished running training LSTMCNN version"

# Third run LSTMPLAIN version
echo "Running training LSTMPLAIN version"
cp config_lstm_plain.ini config.ini
python admire/models/ae_train.py --model_type LSTMPLAIN
echo "Finished running training LSTMPLAIN version"

# Print when finished
echo "-----------------------------"
echo "Finished running training on the remote server"