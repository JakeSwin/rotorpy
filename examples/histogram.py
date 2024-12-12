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
    html.Div([
        dcc.Graph(id='mean'),
        dcc.Graph(id='cov'),
    ], style={'display': 'flex'}),
    dcc.Interval(
        id='interval-component',
        interval=1000
    ),
    dcc.Store(id='data-store', data=np.stack([np.zeros((60, 60)), np.zeros((60, 60))]))
])

@app.callback(
    Output('data-store', 'data'),
    Input('interval-component', 'n_intervals'),
    State('data-store', 'data')
)
def update_data(n, current_data):
    try:
        metadata = subscriber.recv_json(flags=zmq.NOBLOCK)
        msg = subscriber.recv(flags=zmq.NOBLOCK)
        data = np.frombuffer(msg, dtype=metadata["dtype"]).reshape(metadata["shape"])
        return data
    except Exception:
        pass
    return current_data

@app.callback(
    [Output('mean', 'figure'), Output('cov', 'figure')],
    Input('data-store', 'data')
)
def update_heatmap(data):
    fig1 = px.imshow(data[0])
    fig2 = px.imshow(data[1]) 
    return fig1, fig2

if __name__ == "__main__":
    app.run(debug=True)