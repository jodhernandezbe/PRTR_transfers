#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import warnings

from sqlalchemy.sql.expression import true
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

from data_driven.modeling.models import defining_model
from data_driven.modeling.evaluation import prediction_evaluation

from skmultilearn.model_selection import IterativeStratification as KFold
import pandas as pd
import numpy as np
from tqdm import tqdm
import time


def parameter_tuning(X, Y, model, model_params, params_for_tuning):
    '''
    Function to search parameters based on grid search
    '''

    ncycles = sum([len(val) for val in params_for_tuning.values()]) * 5
    pbar = tqdm(desc='Parameter tuning', total=ncycles, initial=0)
    df_tuning = pd.DataFrame(columns=['tuning step', 'param for tuning', 'param value',
                                    'mean validation accuracy', 'mean train accuracy',
                                    'std validation accuracy', 'std train accuracy'])

    step = 0

    for params_key, params_val in params_for_tuning.items():

        step += 1

        for val in params_val:

            kf = KFold(n_splits=5, random_state=None, order=1)
            validation_acc = []
            train_acc = []

            for train_index , validation_index in kf.split(X, Y):

                X_train , X_validation = X[train_index,:], X[validation_index,:]
                Y_train , Y_validation = Y[train_index, :] , Y[validation_index, :]

                model_params.update({params_key: val})
                classifier = defining_model(model, model_params)
                classifier.fit(X_train, Y_train)

                Y_train_hat = classifier.predict(X_train)
                train_acc.append(
                    prediction_evaluation(Y=Y_train, Y_pred=Y_train_hat)
                )

                Y_validation_hat = classifier.predict(X_validation)
                validation_acc.append(
                    prediction_evaluation(Y=Y_validation, Y_pred=Y_validation_hat)
                )

                time.sleep(0.1)
                pbar.update(1)

            mean_train_acc = round(np.mean(np.array(train_acc)), 2)
            mean_validation_acc = round(np.mean(np.array(validation_acc)), 2)
            std_train_acc = round(np.std(np.array(train_acc)), 6)
            std_validation_acc = round(np.std(np.array(validation_acc)), 6)

            df_tuning = pd.concat([df_tuning,
                                    pd.DataFrame({'tuning step': [step],
                                                'param for tuning': [params_key],
                                                'param value': [val],
                                                'mean validation accuracy': [mean_validation_acc],
                                                'mean train accuracy': [mean_train_acc],
                                                'std validation accuracy': [std_validation_acc],
                                                'std train accuracy': [std_train_acc]})
                                ], axis=0, ignore_index=True)

        val_max = df_tuning.loc[df_tuning.loc[df_tuning['param for tuning'] == params_key,
                                              'mean validation accuracy'].idxmax(),
                                'param value']
        model_params.update({params_key: val_max})

    pbar.close()

    classifier = defining_model(model, model_params)
    classifier.fit(X, Y)
    
    model_params = pd.Series(data=model_params.values(), index=model_params.keys())
            

    return classifier, df_tuning, model_params

