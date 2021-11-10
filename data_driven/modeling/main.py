#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.modeling.models import DataDrivenModel
from data_driven.modeling.evaluation import stratified_k_fold_cv

import logging
logging.basicConfig(level=logging.INFO)


def modeling_pipeline(data, model, model_params):
    '''
    Function to run the modeling 
    '''

    logger = logging.getLogger(' Data-driven modeling --> Modeling')

    # Data
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    # Instantiating the model with params
    logger.info(f' Instantiating {model} model')
    classifier = DataDrivenModel(model, **model_params)

    # Cross validation
    logger.info(f' Applying 5-fold cross validation for {model} model')
    score_train, score_validation = stratified_k_fold_cv(model,
                                                    classifier,
                                                    X_train, Y_train)

    return score_train, score_validation