import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import logging
import os

# Initialize logging and plotly
logging.basicConfig(level=logging.DEBUG)
pio.renderers.default = "browser"

# Parameters to change
columns_to_plot = ['power', 'cpu1', 'cpu2']
read_cols = ['hostname', 'date'] + columns_to_plot
filename = '03.2023_tempdata.parquet'
read_path = os.path.join(os.getcwd(), 'data', 'raw', filename)
hosts_to_plot = ['e1122']

# Plot
for hostname in hosts_to_plot:
    
    df = pd.read_parquet(read_path, columns=read_cols)
    
    if hostname not in df['hostname'].unique():
        logging.debug(f"Hostname {hostname} not found")
        continue
    
    df = df[df['hostname'] == hostname] # Filter by hostname
    df['date'] = pd.to_datetime(df['date']) # Convert date to datetime
    
    for column in columns_to_plot:
        fig1 = px.scatter(df, x="date", y=column, color='hostname', color_discrete_map={hostname: 'LightSkyBlue'})
        fig1.update_traces(marker=dict(size=7, color='rgba(45,80,180,0.95)',))
        
        fig2 = px.line(df, x="date", y=column, color='hostname')
        fig2.update_traces(line=dict(color = 'rgba(30,30,30,0.2)', width=0.7))

        fig = go.Figure(data=fig1.data + fig2.data, layout_title_text=f"{hostname} ; {column} ; {filename}")
        fig.show()
