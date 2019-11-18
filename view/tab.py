import dash
import pandas as pd 
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table
import pickle
from lightgbm import LGBMClassifier

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash()
users=pd.read_csv('D:/Dash Final Project/sample_train.csv')
users.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1,inplace=True)
user_book=users[users['country_destination']!='NDF']
user_not=users[users['country_destination']=='NDF']

sessions = pd.read_csv('D:/Dash Final Project/sample_sess.csv')
sessions.drop('Unnamed: 0',axis=1,inplace=True)
sess_book=sessions[sessions['country_destination']!='NDF']
sess_not=sessions[sessions['country_destination']=='NDF']
sessions.drop('country_destination',axis=1,inplace=True)

test_table= pd.read_csv('D:/Dash Final Project/test_final.csv')
test_table.drop('Unnamed: 0',axis=1,inplace=True)

test_model=pd.read_csv('D:/Dash Final Project/test_set.csv')
test_model.set_index('id',inplace=True)

loadModel = pickle.load(open('D:/Dash Final Project/airbnb_recom.sav', 'rb'))

def all_dest():
    table=[{'label':i,'value':i} for i in users['country_destination'].unique()]
    table.append({'label':'All','value':'All'})
    return table

def all_device():
    table=[{'label':i,'value':i} for i in sessions['device_type'].unique()]
    table.append({'label':'All','value':'All'})
    return table

def tab_table():  
    tab=[html.Div(children=[html.Center(html.H2('User dataframe'))]),
         html.Div(children=[
                html.Div(children=[
                    html.P('Country Destination'),
                    dcc.Dropdown(id='destdropdown',
                        options=all_dest(),
                        value='All')],
                        className='col-3'),
                html.Div(children=[html.P('Max Rows'),
                    dcc.Input(id='input-dest-row',type='number',value=5)],
                    className='col-3')],className='row'),
         html.Div(html.Button('Search',id='searchdest')),
         html.Br(),
         html.Div(children=[dash_table.DataTable(id='tabledest',
                                        columns=[{"name": i, "id": i} for i in users.columns],
                                        data=users.to_dict('record'),
                                        page_action='native',
                                        page_current=0,
                                        style_table={'overflowX': 'scroll'},
                                        page_size=5)
                                    ],),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Div(children=[html.Center(html.H2('Session dataframe'))]),
        html.Div(children=[
                html.Div(children=[
                    html.P('Device Type'),
                    dcc.Dropdown(id='devicedropdown',
                        options=all_device(),
                        value='All')],
                        className='col-3'),
                html.Div(children=[html.P('Max Rows'),
                    dcc.Input(id='input-device-row',type='number',value=5)],
                    className='col-3')],className='row'),
         html.Div(html.Button('Search',id='searchdevice')),
         html.Br(),
         html.Div(children=[dash_table.DataTable(id='tabledevice',
                                        columns=[{"name": i, "id": i} for i in sessions.columns],
                                        data=sessions.to_dict('record'),
                                        page_action='native',
                                        page_current=0,
                                        style_table={'overflowX': 'scroll'},
                                        page_size=5)
                                    ],)
                                ]
    return tab


def tab_bar():
    tab=[
        html.Div(children=[
            html.Div(children=[
                html.H5('X1'),
                dcc.Dropdown(id='dropdown_line1',
                    options=[{'label':i,'value':i} for i in users[['date_account_created_new','date_first_active_new','date_first_booking']]],
                    value='date_account_created_new')],
                    className='col-3'),
                ],
                    className='row'),

        html.Div([dcc.Graph(
                    id='graph-scatter',
                    figure={
                        'data':[
                            go.Scatter(
                                x=user_book['date_account_created_new'].value_counts().sort_index().index,  
                                y=user_book['date_account_created_new'].value_counts().sort_index().values,
                                mode='lines',
                                name='user_book'
                                ),
                            go.Scatter(
                                x=user_not['date_account_created_new'].value_counts().sort_index().index,  
                                y=user_not['date_account_created_new'].value_counts().sort_index().values,
                                mode='lines',
                                name='user_book'
                                ),
                            ],

                        'layout':go.Layout(
                            title='AirBnB Accounts Accros Time',
                            xaxis={'title':'Date'},
                            yaxis={'title':'Account'})})
                    ],
                ),
        html.Br(),
        html.Br(),
        html.Div(children=[
            html.Div(children=[
                html.H5('Y1'),
                dcc.Dropdown(id='dropdown',
                    options=[{'label':i,'value':i} for i in users[['gender','signup_method','signup_flow','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser','age_range']]],
                    value='gender')],
                    className='col-3'),],
                    className='row'),
        html.Div([
            dcc.Graph(
                id='graph-bar',
                figure={
                    'data':[
                        {'x':user_book['gender'].value_counts().index,'y':(user_book['gender'].value_counts().values/users.shape[0])*100,'type':'bar','name':'First Booked'},
                        {'x':user_not['gender'].value_counts().index,'y':(user_not['gender'].value_counts().values/users.shape[0])*100,'type':'bar','name':'Not First Booked'}
                        ],
                    'layout':{'title':'Accounts Profile (%)'}
                        })
                ]),
        html.Div([
                dcc.Graph(
                    id='graph-pie',
                    figure={
                        'data':[
                            go.Pie(labels = [i for i in list(users['country_destination'].value_counts().index)],
                            values =list(users['country_destination'].value_counts().values))],
                            'layout':{'title':'Where did the first bookers go?'}
                            })
                            ],
                        )
    ]
    return tab


def tab_pie():
    tab=[
        html.Div(children=[
            html.Div(children=[
                html.H5('Y2'),
                dcc.Dropdown(id='dropdown_session',
                    options=[{'label':i,'value':i} for i in sess_book[['action','action_type','action_detail','device_type']]],
                    value='device_type')],
                    className='col-3'),],
                    className='row'),
        html.Div([
            dcc.Graph(
                id='graph-bar-session',
                figure={
                    'data':[
                        {'x':sess_book.groupby('device_type')['secs_elapsed'].mean().sort_values(ascending=False).head(15).index,'y':sess_book.groupby('device_type')['secs_elapsed'].mean().sort_values(ascending=False).head(15).values/3600,'type':'bar','name':'First Booked'},
                        {'x':sess_not.groupby('device_type')['secs_elapsed'].mean().sort_values(ascending=False).head(15).index,'y':sess_not.groupby('device_type')['secs_elapsed'].mean().sort_values(ascending=False).head(15).values/3600,'type':'bar','name':'Not First Booked'}
                        ],
                    'layout':go.Layout(
                            title='What are they did in AirBnb platform?',
                            yaxis={'title':'Hour (mean)'})
                        })
                ]),
                    ]
    return tab

def tab_scatter():
    tab=[html.Div(children=[
                html.Div(children=[
                    html.P('ID'),
                    dcc.Dropdown(id='iddropdown',
                        options=[{'label':i,'value':i} for i in test_table['id'].unique()],
                        value='jtl0dijy2j')],
                        className='col-3'),
                html.Div(children=[html.Center(html.H5())],
                        id='result',
                        className='col-9')]
                    ,className='row'),
        html.Div(html.Button('Recommendation',id='predict')),
        html.Br(),
        html.Div(children=[html.Center(html.H5('ID Information'))]),
        html.Div(children=[dash_table.DataTable(id='table_id',
                                columns=[{"name": i, "id": i} for i in test_table.columns],
                                data=test_table[test_table['id']=='vvae4amv11'].to_dict('record'),
                                page_action='native',
                                page_current=0,
                                style_table={'overflowX': 'scroll'},
                                page_size=1)
                        ]),
        html.Br(),
        html.Div(children=[html.Center(html.H5('ID Session'))]),
        html.Div(children=[dash_table.DataTable(id='table_sess_id',
                                columns=[{"name": i, "id": i} for i in sessions.columns],
                                data=sessions[sessions['user_id']=='vvae4amv11'].to_dict('record'),
                                page_action='native',
                                page_current=0,
                                style_table={'overflowX': 'scroll'},
                                page_size=5)])
    ]
    return tab