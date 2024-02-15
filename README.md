## Admire anomaly detection
![Tests passing](https://gitlab.pcss.pl/deti/data-science/admire/admire-anomalydetection/badges/CNN_one_node/pipeline.svg)
![Coverage](https://gitlab.pcss.pl/deti/data-science/admire/admire-anomalydetection/badges/CNN_one_node/coverage.svg)

***

# Usage
The main stuff is in the `admire` folder. In order to run training follow these steps:
1. Run `ae_data_prep.py` to prepare the data for training. This will create a `data/processing` folder with the data in the right format.
    There are some tweaks possible to the data preparation. See the `ae_data_prep.py` file for more details (e.g. number of nodes to consider, source filenames).  

1. Make sure that you have the data in `data` folder. There should be a `train` and `test` folder under `data/processing`.  

1. Before running the training make sure to create `wandb_private.key` file in the main directory. This file should contain just the API key for the Weights and Biases service. This is used for logging the training results.   

1. Run `ae_train.py` to train the model. Training is supported by Weights and Biases, so you can see the results in the web interface.

***

# Training the model 

There are a few scripts prepared for training the model on the HPC system, that aim to ease the entire process.

### Copying the project to HPC system
1. In order to copy the project to the HPC system, you can use the `copy_to_remote.sh` script. It will copy the project to the HPC system. Before using, you need to make sure to chech if `username`, `remote_server` and `eagle_path` variables are set correctly to your HPC account / directory.  

1. If this is the first time you are copying the project to the HPC system, you need to set up the conda environment by running the following bash command:
    ```bash
    conda env create -f environment.yml
    ```
    *Remeber to allocate a node to execute this command, as it requires a lot of memory and cpu. -- also, admins might not like it if you do it on the base access.*

    This will create a new conda environment with all the necessary packages. However, this process might be a bit rocky, as the storage for your user's root folder is very limited and conda sets up a lot of packages and puts them right into your root folder. Theoretically, you can override conda settings to change the tmp cache location, but for me it simply didn't work. 

    My way of solving this issue was to change my home directory in .bashrc to a project location, where I had more space. Then I ran the conda environment creation command and after that I changed the home directory back to the original one. 

1. Then you can run the training script. Inside you can set up some variables, which I believe are self-explanatory.
    ```bash
    sbatch slurm_run_training.sh
    ```


***

# Tips

## ae_dataloader.py
This script contains the dataset class for the autoencoder. It is used by `ae_data_prep.py` and `ae_train.py`. It is also possible to use it for other purposes, e.g. to load the data for anomaly detection.
There is not much to tweak here, but it is possible for example to change the number of nodes to consider.

## ae_eval_model.py
This script is for evaluating and visualising the results of autoencoder *after* training by reading checkpoints from `lightning_logs/ae/<run_info>`. 
Here you can change what plots to show and how to do the evaluation.

## ae_train.py
This script is for training the autoencoder. It is possible to tweak the hyperparameters here. The most important ones are the learning rate, the number of epochs and ENCODER_LAYERS (the number of neurons per node in the encoder layers).
Plots generated after training are saved in `images/training` folder. The number of nodes that are used for training is not a parameter, all the nodes that are in the `data/processed` are used.
