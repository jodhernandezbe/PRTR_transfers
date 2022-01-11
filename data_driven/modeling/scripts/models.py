#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.modeling.scripts.metrics import macro_soft_f1, accuracy

from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf


def defining_model(model, model_params):
    '''
    Function to define the model
    '''

    if model == 'RFC':
        dd_model = RandomForestClassifier(**model_params)
    elif model == 'ANNC':
        dd_model = annclassifier(**{par: val for par, val in model_params.items() if par not in ['epochs', 'batch_size', 'verbose']})
        
    return dd_model


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if logs.get('val_accuracy') >= 0.75:
          self.model.stop_training = True


def annclassifier(units_per_layer, dropout, dropout_rate,
                  hidden_layers_activation, learning_rate,
                  beta_1, beta_2, input_shape, output_shape,
                  classification_type):
    '''
    Function to build the Artificial Neural Network Classifier
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
        if dropout[i]:
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
    if classification_type == 'multi-class classification':
        output_activation = 'softmax'
    else:
        output_activation = 'sigmoid'

    model.add(
        tf.keras.layers.Dense(units=output_shape,
                        activation=output_activation)
    )

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=beta_1,
                beta_2=beta_2,
            )

    # Loss function
    if classification_type == 'multi-model binary classification':
        loss_fn = 'binary_crossentropy'
    elif classification_type == 'multi-label classification':
        loss_fn = macro_soft_f1
    else:
        loss_fn = 'categorical_crossentropy'

    # Metric
    if classification_type == 'multi-label classification':
        metric_fn = accuracy
    else:
        metric_fn = 'accuracy'

    # Compiling the model
    model.compile(optimizer=optimizer, 
            loss=loss_fn,
            metrics=[metric_fn])


    return model


