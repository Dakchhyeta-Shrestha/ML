"""Dash interface adapted from the Assignment 1 car-price form."""
import pickle
from pathlib import Path
import sys

import pandas as pd
from dash import Dash, Input, Output, State, dcc, html

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import make_features


def predict_category(bundle, brand, year, mileage, max_power):
    sample = pd.DataFrame([[max_power, mileage, year, brand]],
                          columns=['max_power', 'mileage', 'year', 'brand'])
    prediction = bundle['model'].predict(make_features(sample, bundle['preprocessing']))
    return int(prediction[0])


def create_app(bundle):
    app = Dash(__name__)
    app.layout = html.Div([
        html.H1('Car Price Category'),
        html.P('Predict which of four selling-price ranges a car belongs to.'),
        html.Label('Brand'),
        dcc.Dropdown(bundle['preprocessing']['brands'], id='brand', placeholder='Select brand'),
        html.Label('Year'),
        dcc.Input(id='year', type='number', placeholder='2017'),
        html.Label('Mileage (kmpl)'),
        dcc.Input(id='mileage', type='number', placeholder='18.5'),
        html.Label('Maximum power (bhp)'),
        dcc.Input(id='max-power', type='number', placeholder='100'),
        html.Button('Predict category', id='predict', n_clicks=0),
        html.Div(id='result', style={'marginTop': '20px', 'fontWeight': 'bold'}),
    ], style={'maxWidth': '600px', 'margin': '40px auto', 'fontFamily': 'Arial',
              'padding': '24px', 'border': '1px solid #ddd', 'borderRadius': '12px',
              'display': 'grid', 'gap': '12px'})

    @app.callback(Output('result', 'children'), Input('predict', 'n_clicks'),
                  State('brand', 'value'), State('year', 'value'),
                  State('mileage', 'value'), State('max-power', 'value'))
    def update(n_clicks, brand, year, mileage, power):
        if not n_clicks:
            return ''
        category = predict_category(bundle, brand, year, mileage, power)
        edges = bundle['preprocessing']['edges']
        ranges = [f'up to {edges[0]:,.0f}', f'{edges[0]:,.0f} to {edges[1]:,.0f}',
                  f'{edges[1]:,.0f} to {edges[2]:,.0f}', f'above {edges[2]:,.0f}']
        return f'Predicted class {category}: {ranges[category]} (dataset price units)'

    return app


if __name__ == '__main__':
    with Path(__file__).with_name('model.pkl').open('rb') as stream:
        bundle = pickle.load(stream)
    create_app(bundle).run(host='0.0.0.0', port=8050)
