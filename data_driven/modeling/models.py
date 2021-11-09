#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

class DataDrivenModel:

    def __init__(self, model, **model_params):
        self.model_params = model_params
        self.model = model
        self.set_model()

    def set_model(self):
        if self.model == 'DTC':
            self.dd_model = DecisionTreeClassifier(**self.model_params)
        elif self.model == 'RFC':
            pass
        elif self.model == 'GBC':
            pass
        elif self.model == 'ANNC':
            pass

    def fit(self, X, Y):
        if self.model in ['DTC', 'RFC', 'GBC']:
            self.dd_model.fit(X, Y)
        else:
            pass

    def predict(self):
        pass