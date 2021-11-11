#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
import numpy as np
from scipy.stats import mannwhitneyu

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
    score_analysis = overfitting_underfitting(
                                        results_skfold['train_score'],
                                        results_skfold['test_score']
                                        )

    return score_train, score_validation, score_analysis


def overfitting_underfitting(score_train, score_test):
    '''
    Funtion to determine overfitting and underfitting

    Conditions:
    
    1. High training accurracy (>= 0.75)
    2. Small gap between accucaries

    The non-parametric hypothesis test: Mann Whitney U Test (Wilcoxon Rank Sum Test)

    H0: The score_train and  score_test are equal
    H1: The score_train and  score_test are not equal

    5% level of significance (i.e., α=0.05)
    '''

    if np.mean(score_train) < 0.75:
        return 'underfitting (high bias)'
    else:
        U1, p = mannwhitneyu(score_train,
                            score_test,
                            method="exact")
        if p < 0.05:
            return 'overfitting (high variance)'
        else:
            return 'neither underfitting nor overfitting'
