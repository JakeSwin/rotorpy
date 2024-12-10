import zmq
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

context = zmq.Context()

subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://localhost:5560")
subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='heatmap'),
    dcc.Interval(
        id='interval-component',
        interval=1000
    ),
    dcc.Store(id='data-store', data=np.zeros((60, 60)))
])

@app.callback(
    Output('data-store', 'data'),
    Input('interval-component', 'n_intervals'),
    State('data-store', 'data')
)
def update_data(n, current_zstar):
    try:
        metadata = subscriber.recv_json(flags=zmq.NOBLOCK)
        msg = subscriber.recv(flags=zmq.NOBLOCK)
        zstar = np.frombuffer(msg, dtype=metadata["dtype"]).reshape(metadata["shape"])
        return zstar
    except Exception:
        pass
    return current_zstar 

@app.callback(
    Output('heatmap', 'figure'),
    Input('data-store', 'data')
)
def update_heatmap(zstar):
    figure = px.imshow(zstar) 
    return figure

if __name__ == "__main__":
    app.run(debug=True)