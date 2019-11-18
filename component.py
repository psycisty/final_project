import dash
import pandas as pd 
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
dfpokemon=pd.read_csv('C:/Users/Asyariati/Downloads/Purwadhika/Pokemon.csv')

app.layout=html.Div(children=[
    html.H1('Ini Component'),
    html.P('created by : Ahmad'),
    html.Div(children=[
        html.Div(children=[
            dcc.Dropdown(id='contoh-dropdown',
                        options=[{'label':i,'value':i} for i in dfpokemon.describe().columns],
                        value='contoh')
                        ],className='col-3')
                        ],className='row')
        ])

if __name__ == '__main__':
    app.run_server(debug=True)