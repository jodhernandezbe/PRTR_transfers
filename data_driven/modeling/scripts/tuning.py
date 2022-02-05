#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

from data_driven.modeling.scripts.models import defining_model
from data_driven.modeling.scripts.metrics import prediction_evaluation

import itertools
from skmultilearn.model_selection import IterativeStratification as KFold
from tune_sklearn import TuneSearchCV
from ray.tune.stopper import TrialPlateauStopper
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer


def parameter_tuning(X, Y, model, model_params, params_for_tuning,
                    classification_type):
    '''
    Function to search parameters based on random grid search

    The Stops the entire experiment when the metric has plateaued for more than the given amount of iterations specified in the patience parameter.
    '''

    #n_total = len(list(itertools.product(*params_for_tuning.values())))
    #n_iterations = int(0.05*n_total)

    if classification_type == 'multi-label classification':
        kf = KFold(n_splits=5, random_state=0, order=1)
    else:
        kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

    if model =='RFC':

        search = TuneSearchCV(
                        defining_model('RFC', model_params),
                        params_for_tuning,
                        early_stopping=True,
                        random_state=32,
                        cv=kf,
                        refit='f1',
                        return_train_score=True,
                        n_trials=20,#n_iterations,
                        verbose=0,
                        scoring={'f1': make_scorer(prediction_evaluation, metric='f1'),
                                'accuracy': make_scorer(prediction_evaluation, metric='accuracy')},
                        n_jobs=4,
                        mode='max',
                        search_optimization='bayesian',
                        max_iters=500,
                        stopper=TrialPlateauStopper('f1',
                                                    std=0.01,
                                                    num_results=10,
                                                    grace_period=10,
                                                    mode='max')
        )

        search.fit(X, Y)

        best_params = search.best_params_
        best_estimator = search.best_estimator_
        best_score = search.best_score_

        return {'best_params': best_params,
                'best_estimator': best_estimator,
                'best_score': best_score}

    else:

        pass

        

