#!/bin/bash


echo "Activating the virtual environment"
source /home/users/ignacys/pl0134-01/project_data/pyvenv/bin/activate # Change this path to your virtual environment
echo "Virtual environment activated"

# --data_dir: General data directory
# --model: Model type
# --run_id: Run ID
# --entity: Wandb entity
# --project: Wandb project
# --model_tag: Model tag
# --data_normalization: Data normalization
# --slide_length: Slide length
# --nodes_count: Number of nodes
# --main_data_dir: General data directory

echo "Running RTDATAHANDLER"

python3 admire/models/RTDataHandler.py \
    --data_dir 'data/processed/turin_demo_top200/history' \
    --model 'LSTMCNN' \
    --run_id 'e_-2024_05_08-14_28_22' \
    --entity 'ignacysteam' \
    --project 'lightning_logs' \
    --model_tag 'v0' \
    --data_normalization True \
    --slide_length 1 \
    --nodes_count 200 \
    --main_data_dir 'data/processed/turin_demo_top200'

echo "Bash script finished running RTDATAHANDLER"