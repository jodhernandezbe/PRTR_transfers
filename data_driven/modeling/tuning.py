#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

from data_driven.modeling.models import defining_model

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, make_scorer

def parameter_tuning(X, Y, model, fixed_params, random_grid):
    '''
    Function to search parameters based on randomized grid search
    '''

    regressor = defining_model(model, fixed_params)

    search = RandomizedSearchCV(regressor, random_grid,
                        n_iter=100, 
                        scoring={'mean_absolute_percentage_error': make_scorer(mean_absolute_percentage_error)},
                        cv=5,
                        random_state=1,
                        return_train_score=True,
                        refit='mean_absolute_percentage_error')

    search.fit(X, Y)

    results = search.cv_results_
    time = search.refit_time_
    best_estimator = search.best_estimator_

    return results, time, best_estimator

