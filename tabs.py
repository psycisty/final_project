import dash
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table
import pickle
import numpy as np
from lightgbm import LGBMClassifier
from dash.dependencies import Output,Input,State
from view.tab import tab_bar,tab_pie,tab_scatter,tab_table



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

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

app.layout=html.Div(children=[(html.H1('AirBnB New User Booking')),(html.P('Created by: Ahmad')),
    html.Div(children=[
        dcc.Tabs(value='tabs',id='tabs-1',children= [
            dcc.Tab(label='Dataframe',id='dataframe',children=tab_table()),
            dcc.Tab(label = 'AirBnB Accounts Profile', id = 'tab-satu',children=tab_bar()),
            # dcc.Tab(label = 'Scatter-Chart', id = 'tab-tiga',children=tab_scatter()),
            dcc.Tab(label = 'Session Info', id ='tab-dua',children=tab_pie()),
            dcc.Tab(label = 'Recommendation Machine', id ='tab-tiga',children=tab_scatter())
            ],
                content_style={'fontFamily':'Arial',
                                'borderBottom':'1px solid #d6d6d6',
                                'borderLeft':'1px solid #d6d6d6',
                                'borderRight':'1px solid #d6d6d6',
                                'padding':'50px'}
                            )
                        ]
                    )
                            ],
            style={'maxWidth':'1200px',
                   'margin':'0 auto'}
                   )

@app.callback(
    Output(component_id='graph-bar',component_property='figure'),
    [Input(component_id='dropdown',component_property='value')
    # ,
    # Input(component_id='dropdown1',component_property='value'),
    # Input(component_id='dropdownX',component_property='value')
    ]
)

def create_graph_bar(x1):
    figure={'data':[
                {'x':user_book[x1].value_counts().index,'y':(user_book[x1].value_counts().values/users.shape[0])*100,'type':'bar','name':'First Book'},
                {'x':user_not[x1].value_counts().index,'y':(user_not[x1].value_counts().values/users.shape[0])*100,'type':'bar','name':'Not First Book'}
                ],
            'layout':{'title':'Account Profile (%)'}
                }
    return figure


@app.callback(
    Output(component_id='graph-bar-session',component_property='figure'),
    [Input(component_id='dropdown_session',component_property='value')
    # ,
    # Input(component_id='dropdown1',component_property='value'),
    # Input(component_id='dropdownX',component_property='value')
    ]
)

def create_graph_bar_session(x1):
    figure={'data':[
                {'x':sess_book.groupby(x1)['secs_elapsed'].mean().sort_values(ascending=False).head(15).index,'y':sess_book.groupby(x1)['secs_elapsed'].mean().sort_values(ascending=False).head(15).values/3600,'type':'bar','name':'First Booked'},
                {'x':sess_not.groupby(x1)['secs_elapsed'].mean().sort_values(ascending=False).head(15).index,'y':sess_not.groupby(x1)['secs_elapsed'].mean().sort_values(ascending=False).head(15).values/3600,'type':'bar','name':'Not First Booked'}
                ],
            'layout':go.Layout(
                            title='What are they did in AirBnb platform?',
                            yaxis={'title':'Hour (mean)'})
                }
    return figure

@app.callback(
    Output(component_id='graph-scatter',component_property='figure'),
    [Input(component_id='dropdown_line1',component_property='value'),
    # Input(component_id='dropdown_line2',component_property='value'),
    # Input(component_id='dropdownX',component_property='value')
    ]
)

def create_line(x1):
    figure={
            'data':[
                go.Scatter(
                    x=user_book[x1].value_counts().sort_index().index,  
                    y=user_book[x1].value_counts().sort_index().values,
                    # text=user_book[x1].value_counts().sort_index().values,
                    mode='lines',
                    name='user_book'
                    ),
                go.Scatter(
                    x=user_not[x1].value_counts().sort_index().index,  
                    y=user_not[x1].value_counts().sort_index().values,
                    # text=user_book[x1].value_counts().sort_index().values,
                    mode='lines',
                    name='user_not_book'
                    ),
                ],

            'layout':go.Layout(
                title='AirBnB Account Accros Time',
                xaxis={'title':'Date'},
                yaxis={'title':'Account'})}
    return figure

@app.callback(
    [Output(component_id='table_id',component_property='data'),
    Output(component_id='table_sess_id',component_property='data'),
    Output(component_id = 'result', component_property = 'children')],
    [Input(component_id = 'predict', component_property = 'n_clicks')],
    [State(component_id='iddropdown',component_property='value'),
    State(component_id='iddropdown',component_property='value'),
    State(component_id='iddropdown',component_property='value')]
)

def predict(n_clicks,x1,x2,x3):
    data=test_table[test_table['id']==x1].to_dict('record')
    data1=sessions[sessions['user_id']==x2].to_dict('record')

    pred_probs=loadModel.predict_proba(np.array(test_model[test_model.index==x3])).reshape(1,-1)
    # pred_probs=loadModel.predict_proba(np.array(test_model['id']==x3).reshape(1,-1))
    dest=[loadModel.classes_.tolist()[k] for k in np.argsort(pred_probs[0])[::-1]][:5]
    pred=[html.Center(html.H5('Hey, you might like going to {} {} {} {} {}'.format(dest[0],dest[1],dest[2],dest[3],dest[4])))]

    return data,data1,pred

    
    # predict_prob=loadModel.predict_proba(np.array([x1,x2,x3,x4]).reshape(1,-1))[0]
    # if x1==0:
    #     pred= [html.Center(html.H5(''))]
    # elif predict_prob[0]>predict_prob[1]:
    #     pred = [html.Center(html.H5('The pokemon is NOT LEGENDARY with probabilty {}'.format(round(predict_prob[0],2))))]
    # else:
    #     pred = [html.Center(html.H5('The pokemon is LEGENDARY with probabilty {}'.format(round(predict_prob[1],2))))]
    # return pred





# @app.callback(
#     Output(component_id='graph-pie',component_property='figure'),
#     [Input(component_id='dropdown2',component_property='value')])

# def create_graph_pie(x1):
#     figure={'data':[
#                 go.Pie(labels = [i for i in list(df['Claim Type'].unique())],
#                 values =[df.groupby('Claim Type').mean()[x1][i] for i in list (df['Claim Type'].unique())],sort=False)],
#                 'layout':{'title':'Mean Pie Chart'}
#                 }
#     return figure

@app.callback(
    [Output(component_id='tabledest',component_property='data'),
    Output(component_id='tabledest',component_property='page_size')],
    [Input(component_id='searchdest',component_property='n_clicks')],
    [State(component_id='destdropdown',component_property='value'),
    State(component_id='input-dest-row',component_property='value')]
)

def user_table(n_clicks,x1,x2):
    if x1=='All':
        data=users.to_dict('record')
    else:
        data=users[users['country_destination']==x1].to_dict('record')
    page_size = x2
    return data,page_size

@app.callback(
    [Output(component_id='tabledevice',component_property='data'),
    Output(component_id='tabledevice',component_property='page_size')],
    [Input(component_id='searchdevice',component_property='n_clicks')],
    [State(component_id='devicedropdown',component_property='value'),
    State(component_id='input-device-row',component_property='value')]
    )


def session_table(n_clicks,x1,x2):
    if x1=='All':
        data=sessions.to_dict('record')
    else:
        data=sessions[sessions['device_type']==x1].to_dict('record')
    page_size = x2
    return data,page_size

if __name__ == '__main__':
    app.run_server(debug=True)