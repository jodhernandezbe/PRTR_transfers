#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

from data_driven.modeling.scripts.models import defining_model
from data_driven.modeling.scripts.evaluation import performing_cross_validation

from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.callbacks import DeltaYStopper, DeadlineStopper, EarlyStopper


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


def parameter_tuning(X, Y, model, model_params, search_space,
                    classification_type, time_to_optimize, n_iter_search,
                    threshold, verbose):
    '''
    Function to search parameters based on Bayesian optimization with Gaussian Processes
    '''

    if model =='RFC':

        @use_named_args(search_space)
        def objective_func(**params):
            '''
            Function to optimize parameters
            '''

            model_params.update(**params)
            mean_validation_f1 = performing_cross_validation('RFC', model_params, X, Y, 
                                                        classification_type, for_tuning=True)

            return - mean_validation_f1


        callback1 = DeltaYStopper(delta=1e-4, n_best=3) # PlateuStopper
        callback2 = DeadlineStopper(total_time=time_to_optimize) # For time budget
        callback3 = StoppingDesired(threshold=threshold, n_best=3) # For desired F1 score
        
        search = gp_minimize(objective_func,
                            search_space,
                            n_calls=n_iter_search,
                            callback=[callback1, callback2, callback3],
                            random_state=42,
                            n_jobs=10,
                            verbose=verbose,
                            initial_point_generator='lhs',
                            n_initial_points=4)

        best_params = {search_space[i].name: search.x[i] for i in range(len(search.x))}
        best_mean_validation_f1 = - search.fun

        return {'best_params': best_params,
                'best_mean_validation_f1': best_mean_validation_f1,
                'search': search}

    else:

        pass

        

