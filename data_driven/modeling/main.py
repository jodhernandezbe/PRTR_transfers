#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.modeling.models import DataDrivenModel
from data_driven.modeling.evaluation import stratified_k_fold_cv, y_randomization

import logging
logging.basicConfig(level=logging.INFO)


def modeling_pipeline(X_train, Y_train, model, model_params, return_model=False):
    '''
    Function to run the modeling 
    '''

    logger = logging.getLogger(' Data-driven modeling --> Modeling')

    # Instantiating the model with params
    logger.info(f' Instantiating {model} model')
    classifier = DataDrivenModel(model, **model_params)

    # Cross validation
    logger.info(f' Applying 5-fold cross validation for {model} model')
    cv_result = stratified_k_fold_cv(classifier, X_train, Y_train)

    # Y randomization
    logger.info(f' Applying Y-Randomization for {model} model')
    y_randomization_error = y_randomization(classifier, X_train, Y_train)
    modeling_results =  {**cv_result, **y_randomization_error}

    if return_model:

        # Fitting the model
        classifier.fit(X_train, Y_train)

        return modeling_results, classifier

    else:
        return modeling_results