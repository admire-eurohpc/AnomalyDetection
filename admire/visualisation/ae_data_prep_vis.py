import pandas as pd
import plotly.express as px
import plotly.io as pio

#pio.renderers.default = "browser"
# render in popout window matplotlib style
pio.renderers.default = "browser"
column = 'power'

for hostname in ['e1122']:
    
    df = pd.read_parquet(f"data/processed/test/{hostname}.parquet")
    df['date'] = pd.to_datetime(df['date'])

    print(df['date'].iloc[5746])


    print(df.info())

    fig = px.scatter(df, x="date", y=column, title=f"{hostname} {column}")
    fig.show()
