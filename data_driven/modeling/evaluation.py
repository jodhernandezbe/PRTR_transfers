#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
import numpy as np

def stratified_k_fold_cv(model, classifier, X, Y):
    '''
    Function to apply stratified k-fold cross validation
    '''

    skfold = StratifiedKFold(n_splits=5,
                            random_state=100,
                            shuffle=True)

    results_skfold = cross_validate(classifier,
                                    X, Y,
                                    cv=skfold,
                                    n_jobs=4,
                                    scoring='balanced_accuracy',
                                    return_train_score=True)
    score_validation = np.mean(results_skfold['test_score'])
    score_train = np.mean(results_skfold['train_score'])

    return score_train, score_validation