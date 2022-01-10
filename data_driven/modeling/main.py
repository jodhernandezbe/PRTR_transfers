#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.modeling.models import defining_model
from data_driven.modeling.evaluation import performing_cross_validation, y_randomization

import logging
logging.basicConfig(level=logging.INFO)


def modeling_pipeline(X_train, Y_train, model, model_params, classification_type, return_model=False):
    '''
    Function to run the modeling 
    '''

    logger = logging.getLogger(' Data-driven modeling --> Modeling')

    # Cross validation
    logger.info(f' Applying 5-fold cross validation for {model} model for {classification_type}')
    cv_result = performing_cross_validation(model, model_params, X_train, Y_train, classification_type)

    # Y randomization
    logger.info(f' Applying Y-Randomization for {model} model for {classification_type}')
    y_randomization_error = y_randomization(model, model_params, X_train, Y_train, classification_type)
    modeling_results =  {**cv_result, **y_randomization_error}

    # Comparing Y randomization results and cv
    if cv_result['mean_validation_0_1_loss_or_error'] < y_randomization_error['y_randomization_mean_0_1_loss_or_error'] - y_randomization_error['y_randomization_std_0_1_loss_or_error']:
        modeling_results.update({'y_randomization_analysis': 'Yes'})
    else:
        modeling_results.update({'y_randomization_analysis': 'No'})

    if return_model:

        # Fitting the model
        classifier = defining_model(model, model_params)
        if model == 'RFC':
            classifier.fit(X_train, Y_train)
        else:
            classifier.fit(X_train, Y_train,
                      batch_size=model_params['batch_size'],
                      verbose=model_params['verbose'],
                      epochs=model_params['epochs'],
                      shuffle=model_params['shuffle'],
                      callbacks=[myCallback()])

        return modeling_results, classifier

    else:
        return modeling_results