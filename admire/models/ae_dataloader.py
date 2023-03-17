import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, target_transform=None):
        # Asumme that data is in only one parquet file
        filename = [x[2][0] for x in os.walk(data_dir)][0]
        self.time_series = pd.read_parquet(os.path.join(data_dir, filename))

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, idx):
        ts = self.time_series.iloc[idx]
        if self.transform:
            ts = self.transform(ts)
        return ts
    

if __name__ == '__main__':
    dataset = TimeSeriesDataset(data_dir="data/processed/")
    print(next(iter(dataset)))