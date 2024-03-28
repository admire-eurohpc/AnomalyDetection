import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import tqdm
import numpy as np
import os
import yaml
import wandb
import json
from datetime import datetime

from .metrics import calculate_z_score, get_nth_percentile, threshold_data_by_value, batch_mean_only_working_nodes, batch_median

class Anomaly:
    id: int
    description: str
    severity: int
    date_start: datetime
    date_end: datetime
    tags: list[str]
    
    def __init__(self, 
            id: int, 
            description: str, 
            severity: int, 
            date_start: datetime | str, 
            date_end: datetime | str, 
            tags: list[str],
            tzinfo,
        ) -> None:
        
        self.id = id
        self.description = description
        self.severity = severity
        self.tags = tags
        
        if isinstance(date_start, str):
            date_start = datetime.strptime(date_start, '%Y-%m-%d %H:%M:%S')
            date_start = date_start.replace(tzinfo=tzinfo)
            self.date_start = date_start
        if isinstance(date_end, str):
            date_end = datetime.strptime(date_end, '%Y-%m-%d %H:%M:%S')
            date_end = date_end.replace(tzinfo=tzinfo)
            self.date_end = date_end

def check_if_rec_data_is_loaded(func):
    def wrapper(self, *args, **kwargs):
        if self.reconstruction_data is None:
            raise ValueError('Reconstruction data is not loaded')
        return func(self, *args, **kwargs)
    return wrapper

class MetricsEvaluator:
    
    def __init__(self, 
                 processed_data_dir: str,
                 wandb_run: wandb.run = None,
                 save_path: str = 'images/tmp',
                 ) -> None:
        '''
        Parameters:
            - processed_data_dir: path to the directory containing the processed data (str)
            - wandb_run: wandb.run  if none, the plots will not be logged to wandb
            - save_path: path to the directory where the plots will be saved (str) 
                By default, the plots will be saved in the 'images/tmp' directory
        '''
        self.wandb_run = wandb_run
        self.processed_data_dir = os.path.join(processed_data_dir, 'test')
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        self.plot_data = None
        self.reconstruction_data = None
        
        
    @check_if_rec_data_is_loaded
    def run(self) -> None:
        
        # Read the processed data from the test directory
        self.cpu_alloc_df, self.nodes_sensor_data = self.read_processed_test_data(self.processed_data_dir, self.reconstruction_data)
        
        # Calculate metrics for plots
        self.plot_data = self.calculate_metrics_for_plots(self.reconstruction_data, self.cpu_alloc_df) 
        
        # Transopse and make sure that index is of type datetime
        self.reconstruction_data_transposed = self.reconstruction_data.transpose()
        self.reconstruction_data_transposed.index = pd.to_datetime(self.reconstruction_data_transposed.index)
        
        # Make sure we have the same number of timesteps for all metrics, i.e., our metrics haven't introduced any errors 
        # specifically, we want to make sure that the number of timesteps in the reconstruction data is the same as the number of timesteps in the plot data
        for k, v in self.plot_data.items():
            print(v['data'].shape)
            assert v['data'].shape[0] == self.reconstruction_data_transposed.shape[0], f'{v["data"].shape[0]} != {self.reconstruction_data_transposed.shape[0]}'
            
        # Plot anomalies per timestep
        self.plot_anomalies_per_timestep(
            self.reconstruction_data_transposed, 
            self.plot_data, 
            self.save_path
        )
        
        self.thresholded_plot_data = self.calculate_windowed_metrics(
            plot_data=self.plot_data,
            reconstruction_data_transposed=self.reconstruction_data_transposed,
        )
        
        # Plot smoothed reconstruction error
        self.plot_smoothed_reconstruction_error(
            self.thresholded_plot_data,
            self.save_path
        )
        
        # Plot threshold sensitivity
        self.plot_threshold_sensitivity(
            self.thresholded_plot_data,
            self.save_path
        )
        
        with open('anomalies.yaml', 'r') as f:
            anomalies = yaml.load(f, yaml.UnsafeLoader)
        anomalies_list = []
        for a in anomalies['anomalies']:
            a['tzinfo'] = self.thresholded_plot_data['index_data'][0].timetz().tzinfo
            anomalies_list.append(Anomaly(**a))
            
        self.anomalies_list = anomalies_list
        
        # Plot detected anomalies and ground truth
        self.plot_detected_anomalies_and_ground_truth(
            anomalies_list=self.anomalies_list,
            thresholded_plot_data=self.thresholded_plot_data,
            threshold_to_use=3,
            save_path=self.save_path
        )

             
    def pass_reconstruction_data(self, reconstruction_data: pd.DataFrame) -> None:
        self.reconstruction_data = reconstruction_data    
    
    def retrieve_reconstruction_data_from_artifact(self, 
                wandb_run: wandb.run,
                version_str: str,
                upstream_wandb_dir: str,
    ) -> pd.DataFrame:
        r_wandb_artifact = wandb_run.use_artifact(
            f'run-{version_str}-reconstruction_error:v0',
            type='run_table'
            )
        
        artifact_dir = r_wandb_artifact.download()
        
        with open(artifact_dir, 'r') as f :
            d = json.load(f)

        reconstruction_data = pd.DataFrame(np.array(d['data']), columns=d['columns'])
        
        return reconstruction_data
     
    def calculate_metrics_for_plots(self, reconstruction_data: pd.DataFrame, cpu_alloc_df: pd.DataFrame) -> dict: 
        
        np_rec_error = reconstruction_data.to_numpy()
        plot_data = {}

        # Calculate mean median metrics
        median_np_rec_error = batch_median(np_rec_error)
        mean_filtered_np_rec_error = batch_mean_only_working_nodes(np_rec_error, cpu_alloc_df.to_numpy())
        mean_np_rec_error = np.mean(np_rec_error, axis=0)

        plot_data['median_rec_error'] = {'data': median_np_rec_error, 'name': 'Median reconstruction error'}
        plot_data['mean_filtered_rec_error'] = {'data': mean_filtered_np_rec_error, 'name': 'Mean reconstruction error (mean only on working nodes)'}
        plot_data['mean_rec_error'] = {'data': mean_np_rec_error, 'name': 'Mean reconstruction error (mean on all nodes)'}
        
        # Calculate other metrics 
        z_scores = calculate_z_score(np_rec_error)
        percentiles95 = get_nth_percentile(np_rec_error, 95)
        percentiles90 = get_nth_percentile(np_rec_error, 90)
        percentiles80 = get_nth_percentile(np_rec_error, 80)
        
        # plot_data['z_scores'] = {'data': z_scores, 'name': 'Reconstruction error z-scores'}
        # plot_data['percentiles95'] = {'data': percentiles95, 'name': 'Reconstruction error 95th percentile'}
        # plot_data['percentiles90'] = {'data': percentiles90, 'name': 'Reconstruction error 90th percentile'}
        # plot_data['percentiles80'] = {'data': percentiles80, 'name': 'Reconstruction error 80th percentile'}

        # Threshold data
        thresh_z_scores2 = threshold_data_by_value(z_scores, 2)
        thresh_z_scores3 = threshold_data_by_value(z_scores, 3)
        thresh_percentiles95 = threshold_data_by_value(np_rec_error, percentiles95)
        thresh_percentiles90 = threshold_data_by_value(np_rec_error, percentiles90)
        thresh_percentiles80 = threshold_data_by_value(np_rec_error, percentiles80)
        
        # plot_data['thresh_z_scores2'] = {'data': thresh_z_scores2, 'name': 'Reconstruction error z-scores > 2'}
        # plot_data['thresh_z_scores3'] = {'data': thresh_z_scores3, 'name': 'Reconstruction error z-scores > 3'}
        
        # plot_data['thresh_percentiles95'] = {'data': thresh_percentiles95, 'name': 'Reconstruction error > 95th percentile'}
        # plot_data['thresh_percentiles90'] = {'data': thresh_percentiles90, 'name': 'Reconstruction error > 90th percentile'}
        # plot_data['thresh_percentiles80'] = {'data': thresh_percentiles80, 'name': 'Reconstruction error > 80th percentile'}

        # Sum up the number of anomalies per timestep
        sum_thresh_z_scores2 = np.sum(thresh_z_scores2, axis=0)
        sum_thresh_z_scores3 = np.sum(thresh_z_scores3, axis=0)
        sum_thresh_percentiles95 = np.sum(thresh_percentiles95, axis=0)
        sum_thresh_percentiles90 = np.sum(thresh_percentiles90, axis=0)
        sum_thresh_percentiles80 = np.sum(thresh_percentiles80, axis=0)
        
        plot_data['sum_thresh_z_scores2'] = {'data': sum_thresh_z_scores2, 'name': 'Count of nodes having rec_error\'s z-score > 2'}
        plot_data['sum_thresh_z_scores3'] = {'data': sum_thresh_z_scores3, 'name': 'Count of nodes having rec_error\'s z-score > 3'}
        plot_data['sum_thresh_percentiles95'] = {'data': sum_thresh_percentiles95, 'name': 'Count of nodes having rec_error > 95th percentile'}
        plot_data['sum_thresh_percentiles90'] = {'data': sum_thresh_percentiles90, 'name': 'Count of nodes having rec_error > 90th percentile'}
        plot_data['sum_thresh_percentiles80'] = {'data': sum_thresh_percentiles80, 'name': 'Count of nodes having rec_error > 80th percentile'}
        
        return plot_data

    def read_processed_test_data(self, 
                                 processed_data_dir: str, 
                                 reconstruction_data: pd.DataFrame
        ) -> tuple[pd.DataFrame, dict]:
        # Get all the files in the processed data directory
        files = os.listdir(processed_data_dir)
        print(files)

        columns = None
        cpu_alloc_df = list()
        index_read = list()
        all_data = list()

        for i, file in tqdm.tqdm(enumerate(files), desc='Reading files', total=len(files)):
            if file.endswith('.parquet'):
                d = pd.read_parquet(os.path.join(processed_data_dir, file))
            
                # Get the columns of the data
                if columns is None: 
                    columns = d.columns
                    
                index_read.append(file.replace('.parquet', ''))
                cpu_alloc_df.append(d['cpus_alloc'].astype(int))
                all_data.append(d)
                
        nodes_sensor_data = {k: v for k, v in zip(index_read, all_data)}
                
        cpu_alloc_df = pd.DataFrame(cpu_alloc_df, columns=d.index, index=index_read)
        cpu_alloc_df.info()

        # Sort index of cpu_alloc_df
        cpu_alloc_df.sort_index(inplace=True)

        # Make sure that the index of the reconstruction data is the same as the cpu_alloc_df
        assert reconstruction_data.index.equals(cpu_alloc_df.index)

        # Cut cpu_alloc_df horizontal length to match the reconstruction data
        cpu_alloc_df = cpu_alloc_df.iloc[:, :reconstruction_data.shape[1]]

        # Make sure that horizontal length of the reconstruction data is the same as the cpu_alloc_df
        assert reconstruction_data.shape[1] == cpu_alloc_df.shape[1], f'{reconstruction_data.shape[1]} != {cpu_alloc_df.shape[1]}'
        
        return cpu_alloc_df, nodes_sensor_data
    
    @staticmethod
    def average_past_window(x, w):
        """
        Average over the past w elements of x
        
        Parameters:
            - x: np.array
                Array of shape (n_timesteps,)
            - w: int
                Window size
        """
        x = np.array(x)
        x_padded = np.pad(x, (w-1, 0), mode='constant', constant_values=np.mean(x))
        
        # Use pandas rolling window to calculate the mean
        x_rolling = pd.Series(x_padded).rolling(w).mean().values
        
        # Remove the padded elements
        x_rolling = x_rolling[w-1:]
        
        return x_rolling
    
    @staticmethod
    def average_past_window_per_node(x, w):
        """
        Average over the past w elements of x per node
        
        Parameters:
            - x: np.array
                Array of shape (n_timesteps, n_nodes)
            - w: int
                Window size
                
        Returns:
            - x_rolling: np.array
                Array of shape (n_timesteps, n_nodes)
        """
        x_rolling = np.zeros_like(x)
        for i in range(x.shape[1]):
            x_rolling[:, i] = MetricsEvaluator.average_past_window(x[:, i], w)
            
        return x_rolling




    @staticmethod
    def calculate_windowed_metrics(
            plot_data: dict,
            reconstruction_data_transposed: pd.DataFrame,
            long_window_size: int = 180,
            short_window_size: int = 30,
            stdev_thresholds_to_use: list = [1, 2, 2.5, 3, 4]
        ) -> list:
        '''
        Calculate windowed metrics
        
        Parameters:
            - plot_data: dict
                Dictionary containing the plot data for each metric
            - reconstruction_data_transposed: pd.DataFrame
                Transposed reconstruction data
            - long_window_size: int
                Size of the long window
            - short_window_size: int
                Size of the short window
            - stdev_thresholds_to_use: list
                List of standard deviations to use for thresholding
                
        Returns:
            - thresholded_plot_data: dict
        '''
        time_series_type = 'median'
        time_series = plot_data['median_rec_error']['data'].copy()
        index_data = reconstruction_data_transposed.index.copy()

        sliding_window_mean_long = MetricsEvaluator.average_past_window(time_series, long_window_size)
        sliding_window_mean_short = MetricsEvaluator.average_past_window(time_series, short_window_size)


        # Amputate the first long_window_size elements
        # sliding_window_mean_long = sliding_window_mean_long[long_window_size:len(sliding_window_mean_long)-long_window_size]
        sliding_window_mean_long = sliding_window_mean_long[:len(sliding_window_mean_long)-long_window_size]
        # Amputate the first short_window_size elements
        # sliding_window_mean_short = sliding_window_mean_short[long_window_size:len(sliding_window_mean_short)-long_window_size]
        sliding_window_mean_short = sliding_window_mean_short[:len(sliding_window_mean_short)-long_window_size]
        # Amputate index to match the sliding window mean
        index_data = index_data[:len(index_data)-long_window_size]

        difference = sliding_window_mean_short - sliding_window_mean_long 

        thresholded_list = list()
        for stdevs in stdev_thresholds_to_use:
            stddev = np.std(difference)
            mean = np.mean(difference)
            threshold = mean + stdevs*stddev


            difference_thresholded = difference.copy()
            difference_thresholded[difference_thresholded < threshold] = 0
            difference_thresholded[difference_thresholded >= threshold] = 1
            
            thresholded_list.append(difference_thresholded)
            
        thresholded_plot_data = {
            'difference': difference,
            'thresholded_list': thresholded_list,
            'index_data': index_data,
            'stdev_thresholds_to_use': stdev_thresholds_to_use,
            'long_window_size': long_window_size,
            'short_window_size': short_window_size,
            'time_series_type': time_series_type,
            'index_data': index_data,
            'sliding_window_mean_long': sliding_window_mean_long,
            'sliding_window_mean_short': sliding_window_mean_short,
            'time_series': time_series,
        }
        
        return thresholded_plot_data
    
    def plot_anomalies_per_timestep(self,
                                    reconstruction_data_transposed: pd.DataFrame,
                                    plot_data: dict,
                                    save_path: str,
        ) -> None:
        '''
        Plot the number of anomalies per timestep for each metric in separate row,
        
        Parameters:
            - reconstruction_data_transposed: pd.DataFrame
                Transposed reconstruction data
            - plot_data: dict
                Dictionary containing the plot data for each metric
        '''
        # Plot the number of anomalies per timestep for each metric in separate row, 
        # the first row is the reconstruction error, the last row is average of all sum_thresh metrics
        subplot_titles = [n['name'] for n in plot_data.values()]

        fig = make_subplots(rows=len(plot_data), cols=1, 
                            shared_xaxes=True, 
                            subplot_titles=(subplot_titles)
                            )

        for i, (_, metric_data) in enumerate(plot_data.items()):
            fig.add_trace(go.Scatter(x=reconstruction_data_transposed.index, y=metric_data['data'], name=metric_data['name']), row=i+1, col=1)

        fig.update_layout(height=len(plot_data)*300,
                        title_text="Number of anomalies per timestep for each metric on all nodes aggregated (aggregation type differs by metric e.g., sum, average etc.)",
                        title_x=0.5,
                        )


        title = 'number_of_anomalies_per_timestep'
        filepath = os.path.join(save_path, title + '.html')
        fig.write_html(filepath)
        self.wandb_log_images(title, filepath)
        
    def plot_smoothed_reconstruction_error(self,
                                           thresholded_plot_data: dict,
                                           save_path: str,
        ) -> None:
        '''
        Plot the smoothed reconstruction error in window of long_window_size size.
        
        Parameters:
            - thresholded_plot_data: dict
                Dictionary containing the thresholded plot data
            - save_path: str
                Path to the directory where the plots will be saved
        '''
        
        difference = thresholded_plot_data['difference']
        time_series = thresholded_plot_data['time_series']
        index_data = thresholded_plot_data['index_data']
        sliding_window_mean_long = thresholded_plot_data['sliding_window_mean_long']
        sliding_window_mean_short = thresholded_plot_data['sliding_window_mean_short']
        long_window_size = thresholded_plot_data['long_window_size']
        short_window_size = thresholded_plot_data['short_window_size']
        time_series_type = thresholded_plot_data['time_series_type']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=index_data, y=time_series, name=f'{time_series_type} reconstruction error'))
        fig.add_trace(go.Scatter(x=index_data, y=sliding_window_mean_long, name=f'Sliding window mean (window size={long_window_size})'))
        fig.add_trace(go.Scatter(x=index_data, y=sliding_window_mean_short, name=f'Sliding window mean (window size={short_window_size})'))
        fig.add_trace(go.Scatter(x=index_data, y=difference, name=f'Sliding window mean long short difference'))

        fig.update_layout(height=600,
                            title_text=f"Smoothed reconstruction error in window of {long_window_size} size. \
                                {time_series_type} reconstruction error and sliding window mean",
                            title_x=0.5,
                            )
        filepath = os.path.join(save_path,  f'smoothed_reconstruction_error.html')
        fig.write_html(filepath)
        
        self.wandb_log_images('smoothed_reconstruction_error', filepath)
    
    def plot_threshold_sensitivity(self,
            thresholded_plot_data: dict,
            save_path: str,
        ) -> None:
        '''
        Plot the threshold sensitivity for the smoothed reconstruction error.
        
        Parameters:
            - thresholded_plot_data: dict
                Dictionary containing the thresholded plot data
            - save_path: str
                Path to the directory where the plots will be saved
        '''
        
        difference = thresholded_plot_data['difference']
        time_series = thresholded_plot_data['time_series']
        index_data = thresholded_plot_data['index_data']
        thresholded_list = thresholded_plot_data['thresholded_list']
        time_series_type = thresholded_plot_data['time_series_type']
        STDEV_THRESHOLDS = thresholded_plot_data['stdev_thresholds_to_use']
        LONG_WINDOW_SIZE = thresholded_plot_data['long_window_size']
        SHORT_WINDOW_SIZE = thresholded_plot_data['short_window_size']
          
    
        # Create subplots with three rows and one column
        fig = make_subplots(rows=len(thresholded_list) + 2, cols=1, shared_xaxes=True)

        # Add the first plot - mean reconstruction error
        fig.add_trace(go.Scatter(x=index_data, y=time_series, name=f'{time_series_type} reconstruction error'), row=1, col=1)

        # Add the second plot - difference between last 6 hours mean and last hour mean
        fig.add_trace(go.Scatter(x=index_data, y=difference, name='Absolute difference'), row=2, col=1)  

        for i, thresholded in enumerate(thresholded_list):
            fig.add_trace(go.Scatter(x=index_data, y=thresholded, name=f'Thresholded on (mean * {STDEV_THRESHOLDS[i]} standard deviations)'), row=2 + i+1, col=1)


        # Update layout
        fig.update_layout(height=1200, 
                        title_text=f'{time_series_type} Reconstruction Error and absolute difference between the mean in the last {LONG_WINDOW_SIZE} minutes and last {SHORT_WINDOW_SIZE} minutes.',
                        title_x=0.5)

        filepath = os.path.join(save_path, f'threshold_sensitivity.html')
        fig.write_html(filepath)
        
        self.wandb_log_images('threshold_sensitivity', filepath)

    def plot_detected_anomalies_and_ground_truth(self, 
                                                 anomalies_list: list[Anomaly],
                                                 thresholded_plot_data: dict,
                                                 threshold_to_use: float,
                                                 save_path: str,
        ) -> None:
        '''
        Plot the detected anomalies and the ground truth.
        
        Parameters:
            - anomalies_list: list[Anomaly]
                List of anomalies
            - thresholded_plot_data: dict
                Dictionary containing the thresholded plot data
            - threshold_to_use: float
                Threshold to use for the detected anomalies. Must be one of the thresholds used in the thresholded_plot_data
            - save_path: str
                Path to the directory where the plots will be saved
        '''
        
        time_series_type = thresholded_plot_data['time_series_type']
        time_series = thresholded_plot_data['time_series']
        index_data = thresholded_plot_data['index_data']
        thresholded_list = thresholded_plot_data['thresholded_list']
        long_window_size = thresholded_plot_data['long_window_size']
        short_window_size = thresholded_plot_data['short_window_size']
        thresholds = thresholded_plot_data['stdev_thresholds_to_use']
        
        assert threshold_to_use in thresholds, f'{threshold_to_use} not in {thresholds}'
        
        # Find the indices where data equals 1
        indices = np.where(thresholded_list[threshold_to_use] == 1)[0]
        # Initialize a list to store the regions
        continous_regions = []

        i = 0
        while i < len(indices):
            start = indices[i]
            end = start
            
            # Find the end of the continous region
            while i < len(indices) - 1 and indices[i] + 1 == indices[i+1]:
                end = indices[i+1]
                i += 1
            
            continous_regions.append((start, end))
            print(f'Found continous region from {start} to {end}')
            i += 1
            
        print(f'Found {len(continous_regions)} continous regions')
        
        time_series_std = self.reconstruction_data.std(axis=0)
        # Draw confidence intervals in form of +- 1 standard deviation
        lower = (time_series - time_series_std).astype(float)
        upper = (time_series + time_series_std).astype(float)


        fig = go.Figure([
            go.Scatter(x=index_data, y=time_series, name=f'{time_series_type} reconstruction error', line=dict(color='black')),
            go.Scatter(x=index_data, y=upper, name=f'1 stdev', fill='tonexty', mode='lines', line=dict(width=0), showlegend=False, fillcolor='rgba(200, 200, 200, 0.3)'),
            go.Scatter(x=index_data, y=lower, name=f'+- 1 stdev', fill='tonexty', mode='lines', line=dict(width=0), showlegend=True, fillcolor='rgba(68, 68, 68, 0.3)')
        ])


        for i, (start, end) in enumerate(continous_regions):
            # plot red rectangle around the anomaly range
            if start == end:
                end += 1 # Cheat to make sure that the anomaly range is at least 1 timestep long
                
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=index_data[start],
                y0=0,
                x1=index_data[end],
                y1=1,
                fillcolor="red",
                opacity=0.8,
                layer="below",
                line_width=0,
            )
            
        for i, anomaly in enumerate(anomalies_list):
            # plot red rectangle around the anomaly range
            if anomaly.date_start >= index_data[0] and anomaly.date_end <= index_data[-1]:    
                fig.add_shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=anomaly.date_start,
                    y0=0,
                    x1=anomaly.date_end,
                    y1=1,
                    fillcolor="orange",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                )
            
        # Add legend for the anomaly
        fig.add_trace(go.Scatter(x=[None], y=[None], name=f'Anomaly ground truth', line=dict(color='orange'), mode='lines'))
        # Add legend for the detected anomaly
        fig.add_trace(go.Scatter(x=[None], 
                                 y=[None], 
                                 name=f'Detected anomaly (l:{long_window_size},s:{short_window_size},th:{threshold_to_use})', 
                                 line=dict(color='red'), 
                                 mode='lines')
            )
            
        fig.update_layout(height=600,
                            title_text=f"{time_series_type} reconstruction error and detected anomalies",
                            title_x=0.5,
                            )

        filepath = os.path.join(save_path,  f'detected_anomalies_and_ground_truth.html')
        fig.write_html(filepath)
        
        self.wandb_log_images('detected_anomalies_and_ground_truth', filepath)

    
          
    def wandb_log_images(self, title: str, filepath: str) -> None:
        '''
        Log images to wandb
        
        Parameters:
            - title: str
                Title of the image
            - filepath: str
                Path to the image
        '''
        self.wandb_run.log({title: wandb.Html(open(filepath), inject=False)})

        
if __name__ == '__main__':
    s = '/home/ignacys/Desktop/admire/artifacts/run-e___2024_02_12-14_13_27-reconstruction_error:v0/reconstruction_error.table.json'
    
    with open(s, 'r') as f :
        d = json.load(f)
        
    df = pd.DataFrame(np.array(d['data']), columns=d['columns'])
    print(np.array(d['data']).shape)
    print(df.info())
    print(df.head())
        