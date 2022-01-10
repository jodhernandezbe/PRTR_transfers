#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from random import choices
from data_driven.data_preparation.initial_preprocessing import initial_data_preprocessing
from data_driven.data_preparation.preprocessing import data_preprocessing

import logging
import argparse
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path
logging.basicConfig(level=logging.INFO)

def data_preparation_pipeline(args):
    '''
    Function to run the data preparation pipeline
    '''

    logger = logging.getLogger(' Data-driven modeling --> Data preparation')

    # Preliminary data preprocessing
    logger.info(' Running preliminary data preprocessing')

    filepath = f'{dir_path}/output/data/raw/initial_dataset_{args.id}.csv'    
    if not os.path.isfile(filepath):
        df_ml = initial_data_preprocessing(logger, args)
        df_ml.to_csv(filepath, index=False, sep=',')
    else:
        df_ml = pd.read_csv(filepath)

    # Preprocessing
    logger.info(' Running data preprocessing')

    x_train_path = f'{dir_path}/output/data/transformed/X_train_{args.id}.npy'
    y_train_path = f'{dir_path}/output/data/transformed/Y_train_{args.id}.npy'
    x_test_path = f'{dir_path}/output/data/transformed/X_test_{args.id}.npy'
    y_test_path = f'{dir_path}/output/data/transformed/Y_test_{args.id}.npy'
    
    if not os.path.isfile(x_train_path):
        data = data_preprocessing(df_ml, args, logger)
        np.save(x_train_path, data['X_train'])
        np.save(y_train_path, data['Y_train'])
        np.save(x_test_path, data['X_test'])
        np.save(y_test_path, data['Y_test'])
    else:
        
        X_train = np.load(x_train_path)
        Y_train = np.load(y_train_path)
        X_test = np.load(x_test_path)
        Y_test = np.load(y_test_path)

        if args.classification_type != 'multi-label classification':
            Y_train = Y_train.reshape((Y_train.shape[0], 1))
            Y_test = Y_test.reshape((Y_test.shape[0], 1))

        data = {'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test}
    
    return data

