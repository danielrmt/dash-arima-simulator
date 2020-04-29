# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ALL
from layout_utils import *


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
        marks={i*200:f'{i*200}' for i in range(10)})
]
hyperparam_marks = {i: f'{i}' for i in range(6)}
for param in ['p','d','q']:
    value = 0 if param == 'd' else 1
    sidebar.append(
        html.Div([
            html.Label(param),
            dcc.Slider(id=param, min=0, max=5, step=1, value=value,
                marks=hyperparam_marks)
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
                    min=-1, max=1, step=0.1, value=value, marks=param_marks)
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
def acf_plot_data(acf, mode='markers'):
    return {
        'y': acf,
        'mode': 'markers',
        'error_y': {
            'symmetric': False,
            'array':      np.where(acf < 0, -acf, 0),
            'arrayminus': np.where(acf > 0,  acf, 0)}
        }

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
    plot = {'data': [{'y': y}],
            'layout': {'title': 'Série gerada'}}
    acf_plot  = {'data': [acf_plot_data(acf)],
                 'layout': {'title': 'Função de autocorrelação'}}
    pacf_plot = {'data': [acf_plot_data(pacf)],
                 'layout': {'title': 'Função de autocorrelação parcial'}}
    return plot, acf_plot, pacf_plot


# ----
if __name__ == '__main__':
    app.run_server(debug=True)



