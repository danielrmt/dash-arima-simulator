# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ALL


# ----
app = dash.Dash(
    __name__,
    external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css"]
    )
app.title = "Simulador ARIMA"

server = app.server


# ----
def gen_navbar(brand, items,
    barClass='navbar-dark bg-dark p-0',
    brandClass='col-sm-3 col-md-2 mr-0',
    listClass='px-3',
    itemLiClass='text-nowrap',
    itemAClass=''):
    item_list = []
    for key in items:
        item_list.append(
            html.Li(
                html.A(key, href=items[key],
                    className=f"nav-link {itemAClass}"),
                className=f"nav-item {itemLiClass}"
            )
        )
    return html.Nav(
        [
            html.A(brand, className=f"navbar-brand {brandClass}"),
            html.Ul(item_list, className=f"navbar-nav {listClass}")
        ], className=f"navbar {barClass}"
    )
navbar = gen_navbar('Simulador Arima', {'Github': 'https://github.com/danielrmt/dash-arima-simulator'})



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
sidebar.append(html.Button('Gerar novamente', id='regen'))


# ----
def gen_sidebar_layout(sidebar, content, sidebar_size=2,
    sidebarClass='bg-light', contentClass='', mainClass=''):
    return html.Div(
        [html.Div(sidebar, className=f"sidebar col-md-{sidebar_size} {sidebarClass}"),
         html.Div(content, className=f"col-md-{12-sidebar_size} {contentClass}")],
        className=f"row {mainClass}"
    )



# ----
def gen_grid(items, gridClass='', colClass='', rowClass=''):
    rows = []
    for row in items:
        cols = []
        size = int(12 / len(row))
        for col in row:
            cols.append(html.Div(col, className=f"col-md-{size} {colClass}"))
        rows.append(html.Div(cols, className=f"row {rowClass}"))
    return html.Div(rows, className=f"{gridClass}")

grid = gen_grid([
    [dcc.Graph(id='generated_plot')],
    [dcc.Graph(id='generated_acf'),
     dcc.Graph(id='generated_pacf')]
], "col-md-9")


# ----
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
    plot = {'data': [{'y': y}],
            'layout': {'title': 'Série gerada'}}
    acf  = {'data': [{'y': sm.tsa.stattools.acf(y), 'mode':'markers'}],
            'layout': {'title': 'Função de autocorrelação'}}
    pacf = {'data': [{'y': sm.tsa.stattools.pacf(y), 'mode':'markers'}],
            'layout': {'title': 'Função de autocorrelação parcial'}}
    return plot, acf, pacf


# ----
if __name__ == '__main__':
    app.run_server(debug=True)



