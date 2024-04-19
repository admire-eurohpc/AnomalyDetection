from typing import Sequence
from lightning import LightningModule, Trainer
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
import numpy as np



class CustomWriter(BasePredictionWriter):
    def __init__(self, predictions):
        super().__init__()
        self.pred = predictions
    def on_predict_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: torch.Any, batch: torch.Any, batch_idx: int, dataloader_idx: int=0) -> None:
        print(batch_idx, dataloader_idx)
        self.pred.append(outputs)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        #Possible use of allgather into tensor if data distribution is [0,4,8...], [1,5,9...], [2,6,10...], [3,7,11 ...]. Then we want to gather after each prediction.
        print(np.shape(self.pred))
        return self.pred
        # return super().on_predict_end(trainer, pl_module)
