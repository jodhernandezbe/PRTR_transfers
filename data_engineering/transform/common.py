#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import yaml
import os
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))

file = None


def config(filepath):
    '''
    Function to load yaml files with information needed for transforming the data sources
    '''

    global file
    if not file:
        with open(filepath,
                  mode='r') as f:
            file = yaml.load(f, Loader=yaml.FullLoader)
    return file


def dq_score(system):
    '''
    Function to call the Data Quality Scores
    '''

    dq_path = f'{dir_path}/../../ancillary/DQ_Reliability_Scores.csv'
    dq = pd.read_csv(dq_path, usecols=['source', 'code', 'reliability_score'])
    dq = dq.loc[dq['source'] == system]
    dq_matrix = {row['code']:row['reliability_score'] for idx, row in dq.iterrows()}
    del dq

    return dq_matrix
