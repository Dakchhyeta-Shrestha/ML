import os
import pickle
import numpy as np
import pandas as pd

from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate


# ---------------------------------------------------------
# Load model and preprocessing objects
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')


with open(
    os.path.join(MODEL_DIR, 'model_data.pkl'),
    'rb'
) as file:
    model_data = pickle.load(file)


with open(
    os.path.join(MODEL_DIR, 'scaler.pkl'),
    'rb'
) as file:
    scaler = pickle.load(file)


with open(
    os.path.join(MODEL_DIR, 'brand_encoder.pkl'),
    'rb'
) as file:
    brand_encoder = pickle.load(file)


with open(
    os.path.join(MODEL_DIR, 'imputation_values.pkl'),
    'rb'
) as file:
    imputation_values = pickle.load(file)


theta = model_data['theta']
degree = model_data['degree']


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------

def predict_price(brand, year, mileage, max_power):

    # Handle missing brand
    if brand is None:
        brand_encoded = imputation_values['brand']
    else:
        brand_encoded = brand_encoder.transform([brand])[0]

    # Handle missing numerical values
    if year is None:
        year = imputation_values['year']

    if mileage is None:
        mileage = imputation_values['mileage']

    if max_power is None:
        max_power = imputation_values['max_power']


    # Keep the same feature order used during training
    sample = pd.DataFrame(
        [[max_power, mileage, year, brand_encoded]],
        columns=[
            'max_power',
            'mileage',
            'year',
            'brand'
        ]
    )


    # Apply the training scaler
    sample_scaled = scaler.transform(sample)


    # Build polynomial features:
    # bias + original values + squared values
    features = [
        np.ones((sample_scaled.shape[0], 1)),
        sample_scaled
    ]

    for d in range(2, degree + 1):
        features.append(sample_scaled ** d)

    sample_poly = np.hstack(features)


    # Predict log selling price
    pred_log = sample_poly @ theta

    # Transform back to original selling-price scale
    pred_price = np.exp(pred_log)

    return float(pred_price[0])


# ---------------------------------------------------------
# Dash application
# ---------------------------------------------------------

app = Dash(__name__)

server = app.server


brand_options = [
    {
        'label': brand,
        'value': brand
    }
    for brand in brand_encoder.classes_
]


app.layout = html.Div(

    style={
        'maxWidth': '700px',
        'margin': '40px auto',
        'fontFamily': 'Arial',
        'padding': '30px',
        'border': '1px solid #ddd',
        'borderRadius': '12px'
    },

    children=[

        html.H1(
            'Car Price Prediction',
            style={'textAlign': 'center'}
        ),

        html.P(
            '''
            Enter the available information about the car.
            You may leave a field blank if the information is
            unknown. Missing values will be automatically handled
            before prediction.
            '''
        ),


        html.Label('Brand'),

        dcc.Dropdown(
            id='brand-input',
            options=brand_options,
            placeholder='Select car brand',
            clearable=True
        ),


        html.Br(),

        html.Label('Year'),

        dcc.Input(
            id='year-input',
            type='number',
            placeholder='Example: 2017',
            style={'width': '100%'}
        ),


        html.Br(),
        html.Br(),

        html.Label('Mileage'),

        dcc.Input(
            id='mileage-input',
            type='number',
            placeholder='Example: 18.5',
            style={'width': '100%'}
        ),


        html.Br(),
        html.Br(),

        html.Label('Maximum Power'),

        dcc.Input(
            id='power-input',
            type='number',
            placeholder='Example: 100',
            style={'width': '100%'}
        ),


        html.Br(),
        html.Br(),

        html.Button(
            'Predict Selling Price',
            id='predict-button',
            n_clicks=0,
            style={
                'width': '100%',
                'padding': '12px',
                'fontSize': '16px',
                'cursor': 'pointer'
            }
        ),


        html.Div(
            id='prediction-output',
            style={
                'marginTop': '30px',
                'fontSize': '24px',
                'fontWeight': 'bold',
                'textAlign': 'center'
            }
        )
    ]
)


# ---------------------------------------------------------
# Prediction callback
# ---------------------------------------------------------

@app.callback(
    Output(
        'prediction-output',
        'children'
    ),

    Input(
        'predict-button',
        'n_clicks'
    ),

    State(
        'brand-input',
        'value'
    ),

    State(
        'year-input',
        'value'
    ),

    State(
        'mileage-input',
        'value'
    ),

    State(
        'power-input',
        'value'
    ),

    prevent_initial_call=True
)

def update_prediction(
    n_clicks,
    brand,
    year,
    mileage,
    max_power
):

    if not n_clicks:
        raise PreventUpdate

    try:

        predicted_price = predict_price(
            brand,
            year,
            mileage,
            max_power
        )

        return (
            f'Estimated Selling Price: '
            f'{predicted_price:,.2f}'
        )

    except Exception as error:

        return (
            f'Prediction error: {str(error)}'
        )


if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=8050,
        debug=False
    )
