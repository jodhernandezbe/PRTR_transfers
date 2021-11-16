#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier



def defining_mode(model, model_params):
    '''
    Function to define the model
    '''

    if model == 'DTC':
        dd_model = DecisionTreeClassifier(**model_params)
    elif model == 'RFC':
        dd_model = RandomForestClassifier(**model_params)
    elif model == 'GBC':
        dd_model = XGBClassifier(**model_params)
    elif model == 'ANNC':
        dd_model = MLPClassifier(**model_params)
        
    return dd_model