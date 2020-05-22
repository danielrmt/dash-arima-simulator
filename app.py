# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ALL

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from layout_utils import *


#
pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(
        margin=dict(l=50, r=20, t=40, b=20),
        legend=dict(orientation='h'),
        colorway=["#E69F00", "#56B4E9", "#009E73", "#F0E442", 
                  "#0072B2", "#D55E00", "#CC79A7", "#999999"]
    )
)
pio.templates.default = 'plotly_white+custom'

slider_tooptip = { 'always_visible': True, 'placement': 'right' }


# ----
app = dash.Dash(
    __name__,
    external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css"]
    )
app.title = "Simulador ARIMA"

server = app.server



# ----
sidebar = [
    html.H6('Parâmetros', className='sidebar-heading text-muted'),
    html.Label('Tamanho da série', htmlFor='nsample'),
    dcc.Slider(id='nsample', min=30, max=1000, step=10, value=100,
        marks={i*200:f'{i*200}' for i in range(10)}, tooltip=slider_tooptip)
]
hyperparam_marks = {i: f'{i}' for i in range(6)}
for param in ['p','d','q']:
    value = 0 if param == 'd' else 1
    sidebar.append(
        html.Div([
            html.Label(param),
            dcc.Slider(id=param, min=0, max=5, step=1, value=value,
                marks=hyperparam_marks, tooltip=slider_tooptip)
        ])
    )
sidebar.append(html.Div(id='ar_container'))
sidebar.append(html.Div(id='ma_container'))
sidebar.append(html.Button('Gerar novamente', id='regen', className='btn btn-primary'))

navbar = gen_navbar(app.title,
  {'Github': 'https://github.com/danielrmt/dash-arima-simulator'})

grid = gen_grid([
    [dcc.Graph(id='generated_plot')],
    [dcc.Graph(id='generated_acf'),
     dcc.Graph(id='generated_pacf')]
])

app.layout = html.Div([navbar, gen_sidebar_layout(sidebar, grid, 3)])



# ----
def update_sliders(p, s):
    param_marks = {
        -1: {'label': '-1.0'},
        -0.5: {'label': '-0.5'},
        0: {'label': '0.0'},
        0.5: {'label': '0.5'},
        1: {'label': '1.0'},
    }
    output = []
    for i in range(p):
        value = 0.5 if i == 0 else 0.1
        output.append(
            html.Div([
                html.Label(f"{s}({i+1})", htmlFor=f"{s}{i+1}"),
                dcc.Slider(id={'type':f'{s}-param-slider', 'index': i}, 
                    min=-1, max=1, step=0.1, value=value,
                    marks=param_marks, tooltip=slider_tooptip)
            ])
        )
    return output


@app.callback(
    Output('ar_container', 'children'),
    [Input('p', 'value')])
def update_ar_sliders(p):
    return update_sliders(p, 'AR')


@app.callback(
    Output('ma_container', 'children'),
    [Input('q', 'value')])
def update_ma_sliders(p):
    return update_sliders(p, 'MA')


# ----
def acf_plot_data(acf, title=''):
    return px.scatter(x=np.arange(len(acf)), y=acf, title=title,
        labels={'x':'Lag', 'y':''},
        error_y=np.where(acf < 0, -acf, 0),
        error_y_minus=np.where(acf > 0,  acf, 0))


@app.callback(
    [Output('generated_plot', 'figure'),
     Output('generated_acf', 'figure'),
     Output('generated_pacf', 'figure')],
    [Input('nsample', 'value'),
     Input('d', 'value'),
     Input({'type': 'AR-param-slider', 'index': ALL}, 'value'),
     Input({'type': 'MA-param-slider', 'index': ALL}, 'value'),
     Input('regen', 'n_clicks')])
def generate_data(nsample, d, ar_params, ma_params, regen ):
    ar = [1]
    ma = [1]
    for item in ar_params:
        ar.append(item)
    for item in ma_params:
        ma.append(item)
    ar = -np.array(ar)
    ma = np.array(ma)
    y = sm.tsa.arma_generate_sample(ar, ma, nsample, burnin=1)
    for i in range(d):
        y = np.cumsum(y)
    acf  = sm.tsa.stattools.acf(y)
    pacf = sm.tsa.stattools.pacf(y)
    #
    plot = px.line(x=np.arange(len(y)), y=y, title='Série gerada',
        labels={'x':'', 'y':''})
    acf_plot  = acf_plot_data(acf, 'Função de autocorrelação')
    pacf_plot = acf_plot_data(pacf, 'Função de autocorrelação parcial')
    return plot, acf_plot, pacf_plot


# ----
if __name__ == '__main__':
    app.run_server(debug=True)



