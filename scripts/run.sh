#!/bin/bash


echo "Activating the virtual environment"
source /beegfs/home/Shared/admire/Anomaly_Detection/virtual_env/bin/activate # Change this path to your virtual environment
echo "Virtual environment activated"

cd /beegfs/home/Shared/admire/Anomaly_Detection/AnomalyDetection

ls -la

# --data_dir: General data directory
# --model: Model type
# --run_id: Run ID
# --entity: Wandb entity
# --project: Wandb project
# --model_tag: Model tag
# --data_normalization: Data normalization
# --slide_length: Slide length
# --nodes_count: Number of nodes

echo "Running RTDATAHANDLER"

python3 admire/models/RTDataHandler.py \
    --data_dir '/beegfs/home/Shared/admire/Anomaly_Detection/AnomalyDetection/data/processed/processed/turin_demo_top200' \
    --model 'LSTMCNN' \
    --run_id 'e_-2024_05_08-14_28_22' \
    --entity 'ignacysteam' \
    --project 'lightning_logs' \
    --model_tag 'v0' \
    --data_normalization True \
    --slide_length 1 \
    --nodes_count 66 

