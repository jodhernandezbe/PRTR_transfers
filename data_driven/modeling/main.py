#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.modeling.models import DataDrivenModel

import logging
logging.basicConfig(level=logging.INFO)


def modeling_pipeline(data, model, model_params):
    '''
    Function to run the modeling 
    '''

    logger = logging.getLogger(' Data-driven modeling --> Modeling')

    # Setting the model with params
    classifier = DataDrivenModel(model, **model_params)


    score = None

    return score