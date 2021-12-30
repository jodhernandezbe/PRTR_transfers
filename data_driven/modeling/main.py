#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.modeling.models import defining_model
from data_driven.modeling.evaluation import performing_cross_validation, y_randomization

import logging
logging.basicConfig(level=logging.INFO)


def modeling_pipeline(X_train, Y_train, model, model_params, return_model=False):
    '''
    Function to run the modeling 
    '''

    logger = logging.getLogger(' Data-driven modeling --> Modeling')

    # Cross validation
    logger.info(f' Applying 5-fold cross validation for {model} model')
    cv_result = performing_cross_validation(model, model_params, X_train, Y_train)

    # Y randomization
    logger.info(f' Applying Y-Randomization for {model} model')
    y_randomization_error = y_randomization(model, model_params, X_train, Y_train)
    modeling_results =  {**cv_result, **y_randomization_error}

    if return_model:

        # Fitting the model
        classifier = defining_model(model, model_params)
        classifier.fit(X_train, Y_train)

        return modeling_results, classifier

    else:
        return modeling_results