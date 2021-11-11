#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.modeling.models import DataDrivenModel
from data_driven.modeling.evaluation import stratified_k_fold_cv

import logging
logging.basicConfig(level=logging.INFO)


def modeling_pipeline(X, Y, model, model_params):
    '''
    Function to run the modeling 
    '''

    logger = logging.getLogger(' Data-driven modeling --> Modeling')

    # Instantiating the model with params
    logger.info(f' Instantiating {model} model')
    classifier = DataDrivenModel(model, **model_params)

    # Cross validation
    logger.info(f' Applying 5-fold cross validation for {model} model')
    score_train, score_validation, score_analysis = stratified_k_fold_cv(model,
                                                    classifier,
                                                    X, Y)

    return score_train, score_validation, score_analysis