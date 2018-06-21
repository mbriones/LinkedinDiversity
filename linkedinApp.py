import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from pandas_datareader import data as web
from datetime import datetime as dt
import base64

import flask
import glob
import os

image_directory = '/Users/miguelbriones/Desktop/Insight/LinkedinData/LinkedinImages'
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'

app = dash.Dash()

app.config.supress_callback_exceptions = True

image_filename = 'linkedinlogo.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div(children=[

    html.Div([

             html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                style={'display': 'block', 'margin-left': '80', 'margin-right': '20', 'width': '12%'}),
              ],),
          ], style={'align':'center'}),

    html.Br(),
    html.Br(),

    html.Div([
        dcc.Dropdown(
            id='my-dropdown',
            options=[
                {'label': 'Job ID# 123: Data Scientist I', 'value': 'DS1'},
                {'label': 'Job ID# 456: Data Scientist II', 'value': 'DS2'},
                {'label': 'Job ID# 789: Lead Data Scientist', 'value': 'DS3'},
                {'label': 'Input Own Job Description', 'value': 'DS4'},
                ],
                value='DS3',
                ),
      ], style={'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto', 'width': '40%'}),

    html.Br(),
    html.Br(),

    html.Div([
      dcc.Graph(
        id='my-graph'),
      ]),
])



@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown_value):
    if selected_dropdown_value == 'DS1':
        return {'data': [
                {'x': ['Leadership', 'Tech Savy', 'Academic'], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
            ],
            'layout': {
                'title': 'Employee Traits'}}
    else:
        return {'data': [
                {'x': ['Leadership', 'Tech Savy', 'Academic'], 'y': [1, 4, 3], 'type': 'bar', 'name': 'SF2'},
            ],
            'layout': {
                'title': 'Employee Traits'}}


if __name__ == '__main__':
    app.run_server()
