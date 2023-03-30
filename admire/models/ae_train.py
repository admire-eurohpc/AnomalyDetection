from ae_litmodel import LitAutoEncoder
from ae_dataloader import TimeSeriesDataset
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torch import optim, nn, utils, Tensor
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import logging
import os 
from torchsummary import summary
from datetime import datetime

from ae_encoder import Encoder
from ae_decoder import Decoder

logging.basicConfig(level=logging.DEBUG)

image_save_path = os.path.join('images', 'training')
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

_tmp_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
logger = TensorBoardLogger(save_dir="lightning_logs", name="ae", version=f'{_tmp_name}')

# Setting the seed
L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


LATENT_DIM = 32
BATCH_SIZE = 64
MAX_EPOCHS = 150
SHUFFLE = True
ENCODER_LAYERS = [256, 64]
DECODER_LAYERS = [64, 256]


def calculate_linear_layer_size_after_conv(input_shape, kernel_size, stride, padding):
    '''
    Calculates the size of the linear layer after a convolutional layer.
    
    input_shape: tuple of ints (channels, height, width)
    '''
    channels, height, width = input_shape
    height = (height - kernel_size + 2*padding) / stride + 1
    width = (width - kernel_size + 2*padding) / stride + 1
    return int(channels * height * width)



if __name__ == "__main__":
    
    # Setup data generator class and load it into a pytorch dataloader
    dataset = TimeSeriesDataset(data_dir="data/processed/", normalize=True)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE
        )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE
        )
    
    
    input_size = dataset.get_input_layer_size_flattened()

    input_shape = dataset.get_input_layer_shape()
    channels = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    
    logging.debug(f"input_shape: {input_shape}")
    logging.debug(f'channels: {channels}, height: {height}, width: {width}')

    # BUILD ENCODER
    encoder = nn.Sequential()
    encoder.append(nn.Linear(channels * width * height, ENCODER_LAYERS[0]))
    encoder.append(nn.GELU())
    
    for i in range(1, len(ENCODER_LAYERS)):
        encoder.append(nn.Linear(ENCODER_LAYERS[i-1], ENCODER_LAYERS[i]))
        encoder.append(nn.GELU())
        
    encoder.append(nn.Linear(ENCODER_LAYERS[-1], LATENT_DIM))
    logging.debug(f'Encoder Summary: {encoder}')
    
    # BUILD DECODER
    decoder = nn.Sequential()
    decoder.append(nn.Linear(LATENT_DIM, DECODER_LAYERS[0]))
    decoder.append(nn.GELU())

    for i in range(1, len(DECODER_LAYERS)):
        decoder.append(nn.Linear(DECODER_LAYERS[i-1], DECODER_LAYERS[i]))
        decoder.append(nn.GELU())
        
    decoder.append(nn.Linear(DECODER_LAYERS[-1], channels * width * height))
    logging.debug(f'Decoder Summary: {decoder}')

    # Init the autoencoder
    autoencoder = LitAutoEncoder(input_shape, LATENT_DIM, encoder, decoder)
    
    # Add early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=5, 
        verbose=False, 
        mode="min"
        )

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=[
            early_stop_callback,
            ],
        )
    trainer.fit(
        model=autoencoder, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
        )

    print(os.path.join(logger.save_dir, logger.name, logger.version, "checkpoints"))
    # load checkpoint
    path = os.path.join(logger.save_dir, logger.name, logger.version, "checkpoints")
    filename = os.walk(path).__next__()[2][0]
    checkpoint = os.path.join(path, filename)
    

    autoencoder = LitAutoEncoder.load_from_checkpoint(
                                checkpoint, 
                                encoder=encoder, 
                                decoder=decoder,
                                input_shape=input_shape,
                                latent_dim=LATENT_DIM
                                )

    # choose your trained nn.Module
    encoder = autoencoder.encoder
    encoder.eval()

    # embed 4 fake images!
    fast_check_if_it_works = next(iter(val_loader))
    embeddings = decoder(encoder(fast_check_if_it_works))
    print("⚡" * 20, "\nPredictions:\n", embeddings, "\n", "⚡" * 20)
    print("⚡" * 20, "\nOriginal: \n", fast_check_if_it_works, "\n", "⚡" * 20)
    
    
    def plot_embeddings_vs_real(_embeddings, _real):
        '''
        Plots the embeddings vs the real data.
        '''
        import plotly.express as px
        import plotly.io as pio
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        pio.renderers.default = "browser"
        
        _embeddings = _embeddings.reshape(-1, channels, height, width)
        _real = _real.reshape(-1, channels, height, width)
        
        time_dim = [i for i in range(_embeddings.shape[3])]
        
        indices = [0, 1, 2, 3, 10, 11, 12, 13, 14] # 9 random indices
        channel = ['power', 'cpu1', 'cpu2']
        height_node = 0

        for c, channel in enumerate(channel):
            fig = make_subplots(rows=3, cols=3)
            for i, idx in enumerate(indices):
                
                fig.add_scatter(x=time_dim, y=_embeddings[idx][c][height_node].astype(float), 
                                mode='lines+markers', name='Reconstructed',  
                                line = dict(color='royalblue', width=4, dash='dash'), row=i % 3 + 1, col=i // 3 + 1)
                
                fig.add_scatter(x=time_dim, y=_real[idx][c][height_node].astype(float), 
                                mode='lines+markers', name='Real',  
                                line = dict(color='red', width=4, dash='dot'), row=i % 3 + 1, col=i // 3 + 1)
                
            fig.update_traces(mode='lines+markers' ,overwrite=True)
            fig.update_traces(marker={'size': 9}, overwrite=True)
            fig.update_layout(
                title=f"Validation set Reconstructed vs Real **{channel}** for node (idx) {height_node}\nCheckpoint: '{checkpoint}'",
            )
            fig.update_layout(
                autosize=False,
                width=1400,
                height=1000,
                overwrite=True
                )
            fig.write_image(os.path.join(image_save_path, f'validation_set_reconstructed_vs_real_{channel}_node_{height_node}.png'))
            fig.show()
            
    plot_embeddings_vs_real(embeddings.detach().numpy(), fast_check_if_it_works.detach().numpy())

    