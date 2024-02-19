#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --job-name=admire
#SBATCH --gres gpu:1
#SBATCH -p tesla
#SBATCH --output=slurm_logs/output-%A.txt
#SBATCH --error=slurm_logs/errors/error-%A.txt

echo "Activating the virtual environment"
source /home/users/ignacys/pl0134-01/project_data/pyvenv/bin/activate
echo "Virtual environment activated"

# First run CNN version
echo "Running evaluation CNN version"
python admire/models/ae_eval_model.py --model_type CNN
echo "Finished running evaluation CNN version"

# Second run LSTMCNN version
echo "Running evaluation LSTMCNN version"
python admire/models/ae_eval_model.py --model_type LSTMCNN
echo "Finished running evaluation LSTMCNN version"

# Third run LSTMPLAIN version
echo "Running evaluation LSTMPLAIN version"
python admire/models/ae_eval_model.py --model_type LSTMPLAIN
echo "Finished running evaluation LSTMPLAIN version"

# Print when finished
echo "-----------------------------"
echo "Finished running evaluation on the remote server"