import pandas as pd
import plotly.express as px
import plotly.io as pio

#pio.renderers.default = "browser"
# render in popout window matplotlib style
pio.renderers.default = "browser"
column = 'power'

non_zero = 0
non_zero_list = []
for hostname in [f'e{i}' for i in range(1000, 1250)]:    
    try:
        df = pd.read_parquet(f"data/processed_with_march/test/{hostname}.parquet")
        has = sum(df['power'] > 0.001) > 0
        if has:
            non_zero += 1
            non_zero_list.append(hostname)
    except:
        continue
    
print(f'non_zero records: {non_zero}')
print(f'non_zero_list: {non_zero_list}')

# PLOT
hostname = 'e1122'
df = pd.read_parquet(f"data/processed_with_march/test/{hostname}.parquet")
df['date'] = pd.to_datetime(df['date'])

print(df.describe())

print(df['date'].iloc[123])


print(df.info())

fig = px.scatter(df, x="date", y=column, title=f"{hostname} {column}")
fig.show()
