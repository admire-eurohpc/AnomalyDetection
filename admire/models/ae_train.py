# -- Python imports --
import logging
import os 
import ast
from datetime import datetime
import pandas as pd
import argparse
import wandb
import pickle

# -- Pytorch imports --
import torch
from torch.utils.data import DataLoader
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.accelerators import CPUAccelerator
from pytorch_lightning.loggers import WandbLogger

# -- Project imports --
from ae_encoder import CNN_encoder, CNN_LSTM_encoder
from ae_decoder import CNN_decoder, CNN_LSTM_decoder
from ae_litmodel import LitAutoEncoder, LSTM_AE
from ae_dataloader import TimeSeriesDataset
from utils.config_reader import read_config
from utils.metrics_evaluation import MetricsEvaluator
from utils.transformations import Transform
from ae_eval_model import run_test
from ae_lstm_vae import LSTMVAE, LSTMEncoder, LSTMDecoder

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='', choices=['CNN', 'LSTMCNN', 'LSTMPLAIN', 'LSTMVAE'], help='Model type to train')
parser.add_argument('--experiment_name', type=str, default='', help='Expeiment name')
parser.add_argument('--config_path', type=str, default='config.ini', help='Path to config file')

if parser.parse_args().model_type is not None:
    MODEL_TYPE = parser.parse_args().model_type
if parser.parse_args().experiment_name is not None:
    experiment_name = parser.parse_args().experiment_name
if parser.parse_args().config_path is not None:
    config_path = parser.parse_args().config_path

config_dict = read_config(config_path)

# PREPROCESSING
NODES_COUNT = int(config_dict['PREPROCESSING']['nodes_count_to_process'])
PROCESSED_DATA_DIR = config_dict['PREPROCESSING']['processed_data_dir']

# TRAINING
SEED = int(config_dict['TRAINING']['SEED'])
SHUFFLE = config_dict['TRAINING']['SHUFFLE'].lower() == 'true'
VAL_SHUFFLE = config_dict['TRAINING']['VAL_SHUFFLE'].lower() == 'true'
TENSORBOARD_LOGGING_PATH = config_dict['TRAINING']['TENSORBOARD_LOGGING_PATH']
IMG_SAVE_DIRNAME = config_dict['TRAINING']['IMG_SAVE_DIRNAME']
if MODEL_TYPE == '':
    MODEL_TYPE = config_dict['TRAINING']['MODEL_TYPE']
ENABLE_CHECKPOINTING = config_dict['TRAINING']['ENABLE_CHECKPOINTING'].lower() == 'true'
SAVE_TOP_K_CHECKPOINTS = int(config_dict['TRAINING']['SAVE_TOP_K'])

# EVALUATION
EVALUATION_BATCH_SIZE = int(config_dict['EVALUATION']['BATCH_SIZE'])

# MODEL
model_parameters_key = f'{MODEL_TYPE}_PARAMETERS'
BATCH_SIZE = int(config_dict[model_parameters_key]['BATCH_SIZE'])
MAX_EPOCHS = int(config_dict[model_parameters_key]['MAX_EPOCHS'])
WINDOW_SIZE = int(config_dict[model_parameters_key]['WINDOW_SIZE'])
TRAIN_SLIDE = int(config_dict[model_parameters_key]['TRAIN_SLIDE'])
TEST_SLIDE = int(config_dict[model_parameters_key]['TEST_SLIDE'])
ENCODER_LAYERS = ast.literal_eval(config_dict[model_parameters_key]['ENCODER_LAYERS'])
DECODER_LAYERS = ast.literal_eval(config_dict[model_parameters_key]['DECODER_LAYERS'])
LATENT_DIM = int(config_dict[model_parameters_key]['LATENT_DIM'])
LR = float(config_dict[model_parameters_key]['LEARNING_RATE'])

# Create directories for logging and saving images
_tmp_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
# logger = TensorBoardLogger(save_dir=TENSORBOARD_LOGGING_PATH, name=f"AE_{MODEL_TYPE}", version=f'e_{experiment_name}__{_tmp_name}')

WANDB_UPSTREAM = 'ignacysteam/lightning_logs'
with open('wandb_private.key', 'r') as f:
    wandb_key = f.readline()
    try: 
        wandb.login(
            key=wandb_key
        )
    except Exception as e:
        raise ValueError(f'Cannot log in to wandb: {e}')
    
run = wandb.init(
    project='lightning_logs',
    name=f"AE_{MODEL_TYPE}",
    id=f'e_{experiment_name}-{_tmp_name}',
)
    
print(f'Wandb config: id: {run.id}, name: {run.name}, entity: {run.entity}')
    
logger = WandbLogger(
    save_dir=TENSORBOARD_LOGGING_PATH,
    name=f"AE_{MODEL_TYPE}",
    version=f'e_{experiment_name}__{_tmp_name}',
    log_model=True
)

logdir = f'{TENSORBOARD_LOGGING_PATH}/AE_{MODEL_TYPE}/e_{experiment_name}__{_tmp_name}'

image_save_path = os.path.join(logdir, IMG_SAVE_DIRNAME)

if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)
    
# Set the seed for reproducibility
L.seed_everything(SEED)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    accelerator = 'gpu' if use_cuda else 'cpu'
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {'num_workers': 4, 'pin_memory': False}
    device_num = 1
    

    # -- LOAD DATA -- #
    dataset = TimeSeriesDataset(data_dir=f"{PROCESSED_DATA_DIR}/train/", 
                                normalize=True, 
                                window_size=WINDOW_SIZE, 
                                slide_length=TRAIN_SLIDE)
    
    data_transform: Transform = dataset.get_transform_object()
    
    test_dataset = TimeSeriesDataset(data_dir=f"{PROCESSED_DATA_DIR}/test/",
                                        normalize=True,
                                        window_size=WINDOW_SIZE,
                                        external_transform=data_transform,
                                        slide_length=TEST_SLIDE)
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        drop_last=True,
        **kwargs,
        )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=VAL_SHUFFLE,
        drop_last=True,
        **kwargs,
        )
    
    eval_dataloader = DataLoader(
        dataset=test_dataset,   
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **kwargs,
        )
    # ---------------- #
    
    # -- DEBUG INFO -- #
    d = next(iter(train_loader))
    input_shape = d.shape
    batch = d.shape[0]
    n_features = d.shape[1]
    win_size = d.shape[2]
    input_shape_tuple = (batch, n_features, win_size)

    logging.debug(f"input_shape: {input_shape}")
    logging.debug(f'batch: {batch}, number of features: {n_features}, window size: {win_size}')
    # ---------------- #
    
    # -- SAVE ALL RELEVANT HYPERPARAMETERS -- #
    hparams = {}
    to_copy = [model_parameters_key, 'TRAINING', 'PREPROCESSING']
    for key in to_copy:
        save_key = key.upper()
        if 'PARAMETERS' in key:
            save_key = 'MODEL_PARAMETERS'
        hparams[save_key] = {}
        for param in config_dict[key]:
            if param.lower() not in hparams[save_key] and param.upper() not in hparams[save_key]:
                hparams[save_key][param] = config_dict[key][param]
    
    hparams['TRAINING']['full_training_logs_dir'] = logdir
    logger.log_hyperparams(hparams)
    
    # SAVE TRANSFORM
    artifact = wandb.Artifact(name="Transform", type="object")
    with artifact.new_file("transform.pkl", mode="wb") as f:
        pickle.dump(data_transform, f)  
    run.log_artifact(artifact)
    # ---------------- #
    
    # -- MODEL -- #
    match MODEL_TYPE:
        case 'CNN':
            # Init the lightning autoencoder
            cnn_encoder = CNN_encoder(kernel_size=3, latent_dim=LATENT_DIM, channels=ENCODER_LAYERS)
            cnn_decoder = CNN_decoder(kernel_size=3, latent_dim=LATENT_DIM, channels=DECODER_LAYERS)
            autoencoder = LitAutoEncoder(cnn_encoder, cnn_decoder, lr=LR)
        case 'LSTMCNN':
            SEQUENCE_LENGTH = int(config_dict[model_parameters_key]['SEQUENCE_LENGTH'])
            LSTM_OUT_DIM = int(config_dict[model_parameters_key]['LSTM_OUT_DIM'])
            LSTM_IN_DIM = int(config_dict[model_parameters_key]['LSTM_IN_DIM'])
            LSTM_HIDDEN_CHANNELS = list(ast.literal_eval(config_dict[model_parameters_key]['LSTM_HIDDEN_CHANNELS']))
            cnn_lstm_encoder = CNN_LSTM_encoder(lstm_input_dim=1, lstm_out_dim=LSTM_OUT_DIM, h_lstm_chan=LSTM_HIDDEN_CHANNELS)
            cnn_lstm_decoder = CNN_LSTM_decoder(lstm_input_dim=LSTM_IN_DIM, lstm_out_dim=1, h_lstm_chan=LSTM_HIDDEN_CHANNELS, seq_len=SEQUENCE_LENGTH)
            autoencoder = LitAutoEncoder(cnn_lstm_encoder, cnn_lstm_decoder, lr=LR)
        case 'LSTMPLAIN':
            HIDDEN_SIZE = int(config_dict[model_parameters_key]['HIDDEN_SIZE'])
            autoencoder = LSTM_AE(window_size=win_size, channels=n_features, hidden_size=HIDDEN_SIZE, latent_size=LATENT_DIM, device=device, lr=LR)
        case 'LSTMVAE':
            CHANNELS = int(config_dict[model_parameters_key]['CHANNELS'])
            HIDDEN_SIZE = int(config_dict[model_parameters_key]['HIDDEN_SIZE'])
            LSTM_LAYERS = int(config_dict[model_parameters_key]['LSTM_LAYERS'])
            EMBEDDING_FILTERS = int(config_dict[model_parameters_key]['EMBEDDING_FILTERS'])
            autoencoder = LSTMVAE(
                input_size=WINDOW_SIZE,
                channels=CHANNELS,
                hidden_size=HIDDEN_SIZE,
                latent_size=LATENT_DIM,
                num_lstm_layers=LSTM_LAYERS,
                lr=LR,
                embedding_filters=EMBEDDING_FILTERS,
            )
        case _:
            raise ValueError(f'Invalid model type: {MODEL_TYPE}')
    # ---------------- #
    
    # -- CALLBACKS -- #
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="train_loss", 
        # min_delta=0.0, 
        patience=5, 
        verbose=True, 
        mode="min"
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=SAVE_TOP_K_CHECKPOINTS,
        verbose=True, 
        monitor="train_loss", 
        mode="min"
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval = 'epoch')
    # ---------------- #
    
    # -- PROFILER -- #
    profiler = SimpleProfiler()
    # ---------------- #
    
    # -- TRAINING -- #
    logging.debug(f'Autoencoder Summary: {autoencoder}')
    
    autoencoder.to(device)
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=[
            early_stop_callback,
            checkpoint_callback,
            lr_callback,
            ],   
        enable_checkpointing=ENABLE_CHECKPOINTING,
        accelerator=accelerator,
        devices=device_num, 
        strategy="auto",
        profiler=profiler,
        )
    trainer.fit(
        model=autoencoder, 
        train_dataloaders=train_loader
        )

    logging.debug(os.path.join(logdir, "checkpoints"))
    logging.debug('Training finished.')
    # ---------------- #
    
    # -- TESTING -- #
    logging.debug('Testing...')
    trainer.test(
        model=autoencoder, 
        dataloaders=test_dataloader
    )
    logging.debug('Testing finished.')
    
    
    # -- EVALUATION -- #
    logging.debug('Evaluation...')
    
    test_date_range = test_dataset.get_dates_range()
    test_date_range = pd.date_range(start=test_date_range['start'], end=test_date_range['end'], freq=f'{TEST_SLIDE}min')
    
    save_path = os.path.join(logdir, "eval")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    autoencoder.eval()
    autoencoder.freeze()
    
    rec_error_df = run_test(autoencoder=autoencoder, 
                test_dataloader=eval_dataloader,
                test_dataset=test_dataset,
                test_date_range=test_date_range,
                nodes_count=NODES_COUNT,
                save_rec_err_to_parquet=True,
                test_batch_size=EVALUATION_BATCH_SIZE,
                device=device,
                save_eval_path=save_path,
                wandb_logger=logger
                )
    
    logging.debug('Evaluation finished.')
    
    
    
    metrics_evaluator = MetricsEvaluator(
        processed_data_dir=PROCESSED_DATA_DIR,
        wandb_run=run,
    )
    
    metrics_evaluator.pass_reconstruction_data(rec_error_df)
    
    metrics_evaluator.run()
    
    
    wandb.finish()
    # ---------------- #
    