#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.modeling.scripts.metrics import macro_soft_f1

from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import tensorflow_addons as tfa

def defining_model(model, model_params):
    '''
    Function to define the model
    '''

    if model == 'RFC':
        dd_model = RandomForestClassifier(**model_params)
    elif model == 'ANNC':
        dd_model = annclassifier(**{par: val for par, val in model_params.items() if par not in ['epochs', 'batch_size', 'verbose']})
        
    return dd_model


class StoppingDesiredANN(Callback):

    def __init__(self, threshold=0.70, metric='val_f1'):
        super(Callback, self).__init__()
        self.threshold = threshold
        self.metric = metric

    def on_epoch_end(self, epoch, logs={}):
        if logs.get(self.metric) >= self.threshold:
            self.model.stop_training = True


def annclassifier(hp, units_per_layer, dropout, dropout_rate,
                  hidden_layers_activation, learning_rate,
                  beta_1, beta_2, input_shape, output_shape,
                  classification_type):
    '''
    Function to build the Artificial Neural Network Classifier
    '''

    if not hp:
        hl_act = hidden_layers_activation
    else:
        hl_act = hp.choice('hidden_layers_activation',
                        ['sigmoid', 'tanh', 'relu'],
                           default='relu')

    # Selecting weights initilizer
    if hl_act == 'sigmoid':
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
    if not hp:
        n_hidden_layers = len(units_per_layer)
    else:
        n_hidden_layers = hp.Int('n_hidden_layers',
                                1, 3, step=1,
                                default=1)

    for i in range(n_hidden_layers):

        if not hp:
            droupout_layer = dropout[i]
        else:
            droupout_layer = hp.Choice(f'dropout_layer_{i + 1}',
                                       [True, False],
                                       default=False)

        if droupout_layer:

            if not hp:
                pass
            else:
                dropout_rate = hp.Float(f'dropout_rate_{i + 1}',
                                        0.1, 0.4, step=0.1,
                                        default=0.2)
            model.add(
                tf.keras.layers.Dropout(dropout_rate, seed=0)
                )

        else:

            pass
        if not hp:
            units=units_per_layer[i]
        else:
            units = hp.Int('units_' + str(i), 32, 512,
                        step=10, default=32)
            
        model.add(
            tf.keras.layers.Dense(units=units,
                                activation=hl_act,
                                kernel_initializer=initializer,
                                kernel_regularizer=tf.keras.regularizers.l2(1e-5))
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
    learning_rate = hp.Float('learning_rate',
                            0.001, 0.01, step=0.001,
                            default=0.001)
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
        metric_fn = tfa.metrics.F1Score(threshold=0.5, name='f1', average='macro')
    elif classification_type == 'multi-class classification':
        metric_fn = tfa.metrics.F1Score(threshold=0.5, name='f1', average='micro')
    else:
        metric_fn = tfa.metrics.F1Score(threshold=0.5, name='f1', average='weighted')

    # Compiling the model
    model.compile(optimizer=optimizer, 
            loss=loss_fn,
            metrics=[metric_fn])


    return model


