import pandas as pd
import plotly.express as px
import plotly.io as pio

#pio.renderers.default = "browser"
# render in popout window matplotlib style
pio.renderers.default = "browser"

hostname = "e1110"

column = 'power'

df = pd.read_pickle(f"data/processed/{hostname}.pickle")


print(df.info())

fig = px.scatter(df, x="date", y=column, title=f"{hostname} {column}")
fig.show()

# columns = ['cpu1', 'cpu2']
# fig = px.scatter(df, x="date", y=columns, title=f"{hostname} {columns}")
# fig.show()
