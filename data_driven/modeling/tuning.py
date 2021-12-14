#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

from data_driven.modeling.models import defining_model

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, accuracy_score, make_scorer

def parameter_tuning(X, Y, model, fixed_params, random_grid):
    '''
    Function to search parameters based on randomized grid search
    '''

    classifier = defining_model(model, fixed_params)

    search = RandomizedSearchCV(classifier, random_grid,
                        n_iter=100, 
                        scoring={'balanced_accuracy': make_scorer(balanced_accuracy_score),
                                'accuracy': make_scorer(accuracy_score)},
                        cv=5,
                        random_state=1,
                        return_train_score=True,
                        refit='balanced_accuracy')

    search.fit(X, Y)

    results = search.cv_results_
    time = search.refit_time_
    best_estimator = search.best_estimator_

    return results, time, best_estimator

