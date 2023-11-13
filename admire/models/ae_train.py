# -- Python imports --
import logging
import os 
import ast
from datetime import datetime

# -- Pytorch imports --
import torch
from torch.utils.data import DataLoader
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.accelerators import CPUAccelerator

# -- Project imports --
from ae_encoder import CNN_encoder, CNN_LSTM_encoder
from ae_decoder import CNN_decoder, CNN_LSTM_decoder
from ae_litmodel import LitAutoEncoder
from ae_dataloader import TimeSeriesDataset
from utils.config_reader import read_config

config_dict = read_config()

# PREPROCESSING
NODES_COUNT = int(config_dict['PREPROCESSING']['nodes_count_to_process'])
INCLUDE_CPU_ALLOC = config_dict['PREPROCESSING']['with_cpu_alloc'].lower() == 'true'
PROCESSED_DATA_DIR = config_dict['PREPROCESSING']['processed_data_dir']

# TRAINING
SEED = int(config_dict['TRAINING']['SEED'])
SHUFFLE = config_dict['TRAINING']['SHUFFLE'].lower() == 'true'
VAL_SHUFFLE = config_dict['TRAINING']['VAL_SHUFFLE'].lower() == 'true'
TENSORBOARD_LOGGING_PATH = config_dict['TRAINING']['TENSORBOARD_LOGGING_PATH']
IMG_SAVE_DIRNAME = config_dict['TRAINING']['IMG_SAVE_DIRNAME']
MODEL_TYPE = config_dict['TRAINING']['MODEL_TYPE']
ENABLE_CHECKPOINTING = config_dict['TRAINING']['ENABLE_CHECKPOINTING'].lower() == 'true'
SAVE_TOP_K_CHECKPOINTS = int(config_dict['TRAINING']['SAVE_TOP_K'])

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
logger = TensorBoardLogger(save_dir=TENSORBOARD_LOGGING_PATH, name=f"AE_{MODEL_TYPE}", version=f'{_tmp_name}')
image_save_path = os.path.join(logger.log_dir, IMG_SAVE_DIRNAME)

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
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    accelerator = 'gpu' if use_cuda else 'cpu'
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {'num_workers': CPUAccelerator().auto_device_count(), 'pin_memory': False}
    

    # -- LOAD DATA -- #
    dataset = TimeSeriesDataset(data_dir=f"{PROCESSED_DATA_DIR}/train/", 
                                normalize=True, 
                                window_size=WINDOW_SIZE, 
                                slide_length=TRAIN_SLIDE)
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        drop_last=True,
        **kwargs,
        )
    # ---------------- #
    
    # -- DEBUG INFO -- #
    d = next(iter(train_loader))
    input_shape = d.shape
    batch = d.shape[0]
    n_features = d.shape[1]
    win_size = d.shape[2]

    logging.debug(f"input_shape: {input_shape}")
    logging.debug(f'batch: {batch}, number of features: {n_features}, window size: {win_size}')
    # ---------------- #
    
    # -- SAVE ALL RELEVANT HYPERPARAMETERS -- #
    hparams = {}
    to_copy = [model_parameters_key, 'TRAINING', 'PREPROCESSING']
    for key in to_copy:
        hparams[key] = {}
        for param in config_dict[key]:
            if param.lower() not in hparams[key] and param.upper() not in hparams[key]:
                hparams[key][param] = config_dict[key][param]
    logger.log_hyperparams(hparams)
    # ---------------- #
    
    # -- MODEL -- #
    match MODEL_TYPE:
        case 'CNN':
            # Init the lightning autoencoder
            cnn_encoder = CNN_encoder(kernel_size=3, latent_dim=LATENT_DIM, cpu_alloc=INCLUDE_CPU_ALLOC, channels=ENCODER_LAYERS)
            cnn_decoder = CNN_decoder(kernel_size=3, latent_dim=LATENT_DIM, cpu_alloc=INCLUDE_CPU_ALLOC, channels=DECODER_LAYERS)
            autoencoder = LitAutoEncoder(cnn_encoder, cnn_decoder)
        case 'LSTM':
            SEQUENCE_LENGTH = int(config_dict[model_parameters_key]['SEQUENCE_LENGTH'])
            LSTM_OUT_DIM = int(config_dict[model_parameters_key]['LSTM_OUT_DIM'])
            LSTM_IN_DIM = int(config_dict[model_parameters_key]['LSTM_IN_DIM'])
            LSTM_HIDDEN_CHANNELS = list(ast.literal_eval(config_dict[model_parameters_key]['LSTM_HIDDEN_CHANNELS']))
            cnn_lstm_encoder = CNN_LSTM_encoder(lstm_input_dim=1, lstm_out_dim=LSTM_OUT_DIM, h_lstm_chan=LSTM_HIDDEN_CHANNELS, cpu_alloc=INCLUDE_CPU_ALLOC)
            cnn_lstm_decoder = CNN_LSTM_decoder(lstm_input_dim=LSTM_IN_DIM, lstm_out_dim=1, h_lstm_chan=LSTM_HIDDEN_CHANNELS, cpu_alloc=INCLUDE_CPU_ALLOC, seq_len=SEQUENCE_LENGTH)
            autoencoder = LitAutoEncoder(cnn_lstm_encoder, cnn_lstm_decoder)
        case _:
            raise ValueError(f'Invalid model type: {MODEL_TYPE}')
    # ---------------- #
    
    # -- CALLBACKS -- #
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="train_loss", 
        min_delta=0.001, 
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
        devices="auto", 
        strategy="auto",
        profiler=profiler,
        )
    trainer.fit(
        model=autoencoder, 
        train_dataloaders=train_loader
        )

    logging.debug(os.path.join(logger.save_dir, logger.name, logger.version, "checkpoints"))
    # ---------------- #
    
    logging.debug('Training finished.')