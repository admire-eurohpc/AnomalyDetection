#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --job-name=admire
#SBATCH --gres gpu:1
#SBATCH -p tesla
#SBATCH --output=slurm_logs/output-%A.txt
#SBATCH --error=slurm_logs/errors/error-%A.txt

run_name="standard"
conf="config.ini"
models_to_run=(LSTMVAE LSTMCNN)

echo "Activating the virtual environment"
source /home/users/ignacys/pl0134-01/project_data/pyvenv/bin/activate
echo "Virtual environment activated"

# Prepare data
echo "Preparing data"
python admire/preprocessing/ae_data_prep.py --config_filename $conf --augument
echo "Finished preparing data"

# Run benchmark
for model in "${models[@]}"
do 
    echo "Running training $model version"
    python admire/models/ae_train.py --model_type $model --experiment_name $run_name --config_path $conf
    echo "Finished running training $model version"
done

# Print when finished
echo "-----------------------------"
echo "Finished running training on the remote server"
