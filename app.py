# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


# ----
app = dash.Dash(
    __name__,
    external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css"]
    )
app.title = "Simulador ARIMA"

server = app.server


# ----
def gen_navbar(brand, items,
    barClass='navbar-dark bg-dark flex-md-nowrap p-0 shadow',
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
        marks={i*200:f'{i*200}' for i in range(10)}),
    html.Label('d', htmlFor='d'),
    dcc.Slider(id='d', min=0, max=3, step=1, value=0,
        marks={i: f'{i}' for i in range(4)})
]
marks = {
    -1: {'label': '-1.0'},
    -0.5: {'label': '-0.5'},
    0: {'label': '0.0'},
    0.5: {'label': '0.5'},
    1: {'label': '1.0'},
}
for s in ['AR', 'MA']:
    for i in range(4):
        v = 0.5 if i == 0 else 0
        sidebar.append(html.Label(f"{s}({i+1})", htmlFor=f"{s}{i+1}"))
        sidebar.append(dcc.Slider(id=f"{s}{i+1}", min=-1, max=1, step=0.1, 
        value=v, marks=marks))
sidebar.append(
    html.Button('Gerar novamente', id='regen')
)


# ----
app.layout = html.Div([navbar, 
html.Div(
    className="container-fluid row",
    children=[
        html.Div(
            className="col-md-3 bg-light sidebar",
            children=sidebar
        ),
        html.Div(
            className="col-md-9",
            children=[
                html.Div(
                    className='row',
                    children=[
                        html.Div(
                            className="col-md-12",
                            children=[dcc.Graph(id='generated_plot')]
                        )
                    ]
                ),
                html.Div(
                    className='row',
                    children=[
                        html.Div(
                            className="col-md-6",
                            children=[dcc.Graph(id='generated_acf')]
                        ),
                        html.Div(
                            className="col-md-6",
                            children=[dcc.Graph(id='generated_pacf')]
                        )
                    ]
                )
            ]
        )
    ]
)])

# ----
@app.callback(
    [Output('generated_plot', 'figure'),
     Output('generated_acf', 'figure'),
     Output('generated_pacf', 'figure')],
    [Input('nsample', 'value'),
     Input('d', 'value'),
     Input('AR1', 'value'),
     Input('AR2', 'value'),
     Input('AR3', 'value'),
     Input('AR4', 'value'),
     Input('MA1', 'value'),
     Input('MA2', 'value'),
     Input('MA3', 'value'),
     Input('MA4', 'value'),
     Input('regen', 'n_clicks')])
def generate_data(nsample, d, ar1, ar2, ar3, ar4, ma1, ma2, ma3, ma4, regen ):
    ar = -np.array([-1, ar1, ar2, ar3, ar4])
    ma = np.array([1, ma1, ma2, ma3, ma4])
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



