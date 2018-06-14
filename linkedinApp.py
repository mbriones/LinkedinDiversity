import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from pandas_datareader import data as web
from datetime import datetime as dt
import base64

app = dash.Dash()

image_filename = 'google.png'
image_filename2 = 'Twitter1.png'
image_filename3 = 'Twitter2.png'

encoded_image = base64.b64encode(open(image_filename, 'rb').read())
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())
encoded_image3 = base64.b64encode(open(image_filename3, 'rb').read())



app.layout = html.Div([
    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())),
    html.H2('Job Title'),
    dcc.Dropdown(
        id='my-dropdown',
        options=[
            {'label': 'Data Scientist', 'value': 'DS'},
            {'label': 'Data Analyst', 'value': 'DA'}
        ],
        value='DS'
    ),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                #{'x': [1, 2, 3, 4], 'y': [4, 1, 2, 3], 'type': 'bar', 'name': 'SF'},
                {'x': ['Entrepreneurial', 'Tech', 'Mentorship', 'Leadership'], 'y': [2, 4, 5, 3], 'type': 'bar', 'name': u'Topic'},
            ],
            'layout': {
                'title': 'Summary Score'
            }
        }
    ),

    dcc.RadioItems(
        options=[
            {'label': 'Entrepreneurial', 'value': 'ENT'},
            {'label': 'Tech', 'value': 'TEC'},
            {'label': 'Mentorship', 'value': 'MEN'},
            {'label': 'Leadership', 'value': 'LED'}
            ],
            value='LED',
            labelStyle={'display': 'inline-block'}
    ),

    html.H2('Recommended Profiles'),
    html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode())),
    html.Img(src='data:image/png;base64,{}'.format(encoded_image3.decode()))
])

'''
@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    df = web.DataReader(
        selected_dropdown_value, data_source='google',
        start=dt(2017, 1, 1), end=dt.now())
    return {
        'data': [{
            'x': df.index,
            'y': df.Close
        }]
    }
'''

if __name__ == '__main__':
    app.run_server()
