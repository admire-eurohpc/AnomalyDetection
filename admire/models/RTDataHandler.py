import pandas as pd
import numpy as np
import redis
import time
from datetime import datetime
from typing import Dict, List

from timeseriesdatasetv2 import TimeSeriesDatasetv2



class RTDataHandler:
    def __init__(self,) -> None:
        '''
        For simulation purposes we need to check time of invoking RTDataHandler to read/save proper data.
        We don't need year, month, day information since it's only one day of data passing through
        Intentionally omitting seconds, cause you can easily find required data through hour and minute values
        TODO : decide if we want to simulate hour from 00:00 of 23th or if we want to check current hour and assume that history end at current time
        '''
        #For now implementing second option
        current_time = datetime.now()
        self.hour = current_time.hour
        self.minute = current_time.minute
        self.batch_time = 5 #TODO : remove hardcoding

    def connect_to_db(self) -> redis.client.Redis:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        return r
    
    def log_to_db(self,) -> None:
        pass
    
    def get_new_data_from_db(self) -> np.array:
        '''
        Default pipeline for feeding data every 5 minutes.
        TODO: How to ensure reading proper data when script is automatically started every 5 minutes?
        Possibilities : 
        - Cron starting every 5 minutes the bash script with RTDataHandler and passing an argument (internal bash iterator).
        - We cut first data batch at the end of script life
        '''
        db_dataloader = TimeSeriesDatasetv2(
            data_dir='data/processed/turin_demo_top200/valid_data',
            normalize=True,
            window_size=5,
            slide_length=1,
            nodes_count=200
        )
        data_batch = db_dataloader[0].numpy()
        print(np.shape(data_batch))
        return data_batch
    
    def trim_db_data(self):
        db_dataloader = TimeSeriesDatasetv2(
            data_dir='data/processed/turin_demo_top200/valid_data',
            normalize=True,
            window_size=5,
            slide_length=1,
            nodes_count=200
        )
        data_batch = db_dataloader.get_time_series()
        pass


    def get_history(self) -> tuple[np.array, List, Dict[str, datetime]]:
        history_dataloader = TimeSeriesDatasetv2(
            data_dir='data/processed/turin_demo_top200/history',
            normalize=True,
            window_size=60,
            slide_length=1,
            nodes_count=200
        )
        history = history_dataloader.get_time_series()
        node_names = history_dataloader.get_node_names()
        dates_range = history_dataloader.get_dates_range()
        #Setting history dates_range to batch_time minutes before so with addition of current batch we get current time
        # TODO: Technically we need to only to it once at the beginning of demo, let's try to find a better way to implement it
        dates_range['start'] = dates_range['start'].replace(hour=self.hour, minute=self.minute-self.batch_time)
        dates_range['end'] = dates_range['end'].replace(hour=self.hour, minute=self.minute-self.batch_time)
        print(dates_range)
        return history, node_names, dates_range
    
    def save_history(self, history: np.array, node_names: list, dates_range: Dict[str, datetime]) -> pd.DataFrame:
        for elem, filename in zip(history, node_names):
            _df = pd.DataFrame(np.transpose(elem), columns = ['power', 'cpu1', 'cpu2', 'cpus_alloc'])
            _df['date'] = pd.date_range(dates_range['start'], dates_range['end'], freq='1min', tz='UTC')
            #_df.to_feather(os.path.join('data/processed/turin_demo_top200/history_updated', filename + '.feather'))
        print('kkk', np.shape(history))
        pass
    
    def get_metrics_from_db(self, ) -> dict:
        '''
        We could check for consistency of our detection and check against previous detections.
        '''
        pass

    def append_log_file(self, ) -> None:
        '''
        If we want to check against previous detection, we need to update log_file every data batch
        '''
        pass

    def data_checkup(self) -> None:
        '''
        Some tests to run against incoming data? Maybe it's redundant in a demo, but for sure will be needed in real life application.
        '''
        pass

    def caluclate_metrics(self):
        '''
        Metrics Evaluator for short data batches.
        '''
        pass

    def run(self,) -> None:
        #db = self.connect_to_db()
        #db.set('foo', 'bar')
        start = time.time()
        #get database data 
        data_batch = self.get_new_data_from_db()
        #get history data
        history, node_names, dates_range = self.get_history()
        #data on which we want to perform metrics calculation
        history_new = np.concatenate((history, data_batch), axis=2)

        #calculate metrics here
    
        dates_range_new={}
        dates_range_new['start'] = dates_range['start'].replace(hour=self.hour, minute=self.minute)
        dates_range_new['end'] = dates_range['end'].replace(hour=self.hour, minute=self.minute)

        self.save_history(history_new[:,:,self.batch_time:], node_names, dates_range_new)
        print(np.shape(history_new))
        end = time.time()
        print(end-start)

if __name__ == '__main__':
    #For testing purposes feeding date from bash script
    logger = RTDataHandler()
    logger.run()
