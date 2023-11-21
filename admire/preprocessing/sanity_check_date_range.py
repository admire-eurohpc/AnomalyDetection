import pandas as pd
import datetime
import os

DATA_PATH = 'data/processed/march22-24_top200_withalloc_and_augm_fixed_hours_v3'

TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH =  os.path.join(DATA_PATH, 'test')

# Filename of first parquet file in each directory
train_filename = os.listdir(TRAIN_PATH)[0]
test_filename = os.listdir(TEST_PATH)[0]

train_data = pd.read_parquet(os.path.join(TRAIN_PATH, train_filename))
test_data = pd.read_parquet(os.path.join(TEST_PATH, test_filename))

# Check if train contains test data
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])


train_test_overlap = train_data[train_data['date'].isin(test_data['date'])]
overlap_records_count = len(train_test_overlap)

print(f'---  Data date range check  ---')
print(f"Train records count: {len(train_data)}")
print(f"Test records count: {len(test_data)}")
print(f'Train and test data overlap: {overlap_records_count > 0}')
print(f"Overlap records count: {overlap_records_count}")
print(f"Overlap starts at: {train_test_overlap['date'].min()}")
print(f"Overlap ends at: {train_test_overlap['date'].max()}")
print(f"*Note, that if overlap is not continous, start/end dates are incorrect!")
print(f'-'*30)