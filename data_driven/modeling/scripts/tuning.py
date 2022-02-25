#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

from data_driven.modeling.scripts.evaluation import performing_cross_validation

from skopt import gp_minimize
from skopt import load
from skopt.callbacks import DeltaYStopper, DeadlineStopper, EarlyStopper, CheckpointSaver
import os
from functools import partial

'''
Random Forest Classifier
'''

class StoppingDesired(EarlyStopper):
    
    def __init__(self, threshold=0.70, n_best=3):
        super(EarlyStopper, self).__init__()
        self.threshold = threshold
        self.n_best = n_best

    def _criterion(self, result):
        if len(result.func_vals) >= self.n_best:
            return result.func_vals[-1] <= -self.threshold
        else:
            return None

# Defining callbacks
def rfc_parameter_tuning(X, Y, classification_type, model_params,
                    search_space, time_to_optimize, threshold,
                    x_initial, y_initial, n_calls, verbose, n_initial_points,
                    target_class=None):
    '''
    Function to run Bayesian Optimization with Gaussian Processes for RFC
    '''

    if target_class:
        file = f'RFC_class_{target_class}_tuning.pkl'
    else:
        file = 'RFC_tuning.pkl'

    checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    os.pardir,
                                    'output',
                                    'models',
                                    classification_type.replace(' ', '_'),
                                    file
                                    )
    callback1 = DeltaYStopper(delta=1e-4, n_best=5) # PlateuStopper
    callback2 = DeadlineStopper(total_time=time_to_optimize) # For time budget
    callback3 = StoppingDesired(threshold=threshold, n_best=5) # For desired F1 score
    callback4 = CheckpointSaver(checkpoint_path, compress=9) # For saving checkpoints

    if os.path.isfile(checkpoint_path):
        res = load(checkpoint_path)
        x0 = res.x_iters
        y0 = res.func_vals
    else:
        x0 = [x_initial[search_space[i].name] for i in range(len(search_space))]
        y0 = y_initial

    search = gp_minimize(partial(objective_func, model_params=model_params,
                                classification_type=classification_type,
                                X=X, Y=Y,
                                order_keys=[search_space[i].name for i in range(len(search_space))]),
                        search_space,
                        x0=x0,
                        y0=y0,        
                        n_calls=n_calls,
                        callback=[callback1, callback2,
                                    callback3, callback4],
                        random_state=42,
                        verbose=verbose,
                        n_jobs=10,
                        n_initial_points=n_initial_points)

    best_params = {search_space[i].name: search.x[i] for i in range(len(search.x))}
    best_objective = - search.fun

    return {'best_params': best_params,
            'best_objective': best_objective,
            'search': search}


#@use_named_args(search_space)
def objective_func(params, model_params, classification_type, X, Y, order_keys):
    '''
    Function to optimize parameters
    '''

    model_params.update({order_keys[i]: params[i] for i in range(len(params))})
    cv_result = performing_cross_validation('RFC', model_params, X, Y, 
                                                classification_type, for_tuning=True)

    mean_validation_f1 = cv_result['mean_validation_f1']

    return - mean_validation_f1


'''
Artificial Neural Network Classifier
'''







        

