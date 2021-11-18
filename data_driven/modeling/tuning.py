#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.modeling.models import defining_model

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

def parameter_tuning(X, Y, model, fixed_params, space):
    '''
    Function to search parameters based on randomized grid search
    '''

    skfold = StratifiedKFold(n_splits=5,
                            random_state=100,
                            shuffle=True)

    classifier = defining_model(model, **fixed_params)

    search = RandomizedSearchCV(classifier, space,
                        n_iter=100, 
                        scoring=('balanced_accuracy',
                                'accuracy'),
                        n_jobs=-1, cv=skfold,
                        random_state=1,
                        return_train_score=True,
                        refit='balanced_accuracy')

    search.fit(X, Y)

    results = search.cv_results_
    time = search.refit_time_
    best_estimator = search.best_estimator_

    return results, time, best_estimator