#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf


def defining_model(model, model_params):
    '''
    Function to define the model
    '''

    if model == 'DTR':
        dd_model = DecisionTreeRegressor(**model_params)
    elif model == 'RFR':
        dd_model = RandomForestRegressor(**model_params)
    elif model == 'GBR':
        dd_model = MultiOutputRegressor(estimator=XGBRegressor(**model_params))
    elif model == 'ANNR':
        epochs = model_params['epochs']
        batch_size = model_params['batch_size']
        verbose = model_params['verbose']
        model_params = {par: val for par, val in model_params.items() if par not in ['epochs', 'batch_size', 'verbose']}
        dd_model = tf.keras.wrappers.scikit_learn.KerasRegressor(
           build_fn=annregressor,
           **model_params,
           epochs=epochs,
           batch_size=batch_size,
           verbose=verbose
        )
        
    return dd_model


def annregressor(units_per_layer, dropout, dropout_rate,
                  hidden_layers_activation, learning_rate,
                  beta_1, beta_2, input_shape, output_shape):
    '''
    Function to build the Artificial Neural Network Regressor
    '''

    # Selecting weights initilizer
    if hidden_layers_activation == 'sigmoid':
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
    else:
        initializer = tf.keras.initializers.VarianceScaling(seed=0)

    # Initialize model
    model = tf.keras.Sequential()

    # Input layer
    model.add(
        tf.keras.Input(shape=(input_shape,))
                )
    
    # Hidden layers
    n_hidden_layers = len(units_per_layer)
    for i in range(n_hidden_layers):
        if dropout:
            model.add(
                tf.keras.layers.Dropout(dropout_rate, seed=0)
                )
        else:
            pass
        model.add(
            tf.keras.layers.Dense(units=units_per_layer[i],
                                activation=hidden_layers_activation,
                                kernel_initializer=initializer)
        )

    # Output layer
    model.add(
        tf.keras.layers.Dense(units=output_shape,
                        activation=None)
    )

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2,
            )

    # Compiling the model
    model.compile(optimizer=optimizer, 
            loss='mse',
            metrics=['mape'])


    return model


