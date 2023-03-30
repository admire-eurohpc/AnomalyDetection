from typing import List
import numpy.typing as npt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def plot_embeddings_vs_real(_embeddings: npt.NDArray, 
                            _real: npt.NDArray, 
                            channels: int, 
                            height: int, 
                            width: int, 
                            checkpoint: str, 
                            image_save_path: str) -> None:
        '''
        Plots the embeddings vs the real data.
        
        `_embeddings` is a numpy array of shape (batch_size, channels, height, width)
        `_real` is a numpy array of shape (batch_size, channels, height, width)
        `
    
        '''
       
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
            fig.write_image(
                os.path.join(image_save_path, f'validation_set_reconstructed_vs_real_{channel}_node_{height_node}.png')
                )
            fig.show()
            
            
def plot_reconstruction_error_over_time(reconstruction_errors: List[float],
                                        time_axis: List[int] | None = None) -> None:
    '''
    Plots the reconstruction error over time.
    
    `reconstruction_errors` is a list of floats, where each float is the reconstruction error for a single timestep.
    `time_axis` is a list of ints, where each int is the timestep for the corresponding reconstruction error.
    '''
    pio.renderers.default = "browser"
    
    if time_axis is None:
        time_axis = [i for i in range(len(reconstruction_errors))]
    
    fig = make_subplots(rows=1, cols=1)
    fig.add_scatter(x=time_axis, y=reconstruction_errors)
    fig.update_layout(
        title=f"Reconstruction error over time",
    )
    fig.show()