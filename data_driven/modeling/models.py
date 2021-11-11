#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

class DataDrivenModel(BaseEstimator, ClassifierMixin):

    def __init__(self, model, **model_params):
        self.model_params = model_params
        self.model = model
        self._data_driven = self.set_model()

    def set_model(self):
        if self.model == 'DTC':
            dd_model = DecisionTreeClassifier(**self.model_params)
        elif self.model == 'RFC':
            dd_model = RandomForestClassifier(**self.model_params)
        elif self.model == 'GBC':
            dd_model = GradientBoostingClassifier(**self.model_params)
        elif self.model == 'ANNC':
            dd_model = MLPClassifier(**self.model_params)
        return dd_model

    def fit(self, X, Y):
        self._data_driven.fit(X,Y)

    def predict(self, X):
        return self._data_driven.predict(X)

    def predict_proba(self, X):
        return self._data_driven.predict_proba(X)