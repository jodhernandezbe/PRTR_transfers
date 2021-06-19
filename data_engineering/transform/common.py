#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import yaml
import os
import pandas as pd


dir_path = os.path.dirname(os.path.realpath(__file__))


def config(filepath):
    '''
    Function to load yaml files with information needed for transforming the data sources
    '''

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


def opening_files(usecols=['national_substance_name',
                'national_substance_id', 'cas_number'],
                dtype={'national_substance_id': object},
                systems_class=['TRI', 'NPI', 'NPRI'],
                column_name='prtr_system'):
    '''
    Function to open the transformed PRTR files for getting information
    '''

    # Searching for PRTR files
    output_path = f'{dir_path}/output'
    list_of_files = ['tri.csv', 'npi.csv', 'npri.csv']
        
    # Concatenating information from PRTR files
    df = pd.DataFrame()
    for file in list_of_files:
        df_aux = pd.read_csv(f'{output_path}/{file}', usecols=usecols,
                            dtype=dtype)
        df_aux.drop_duplicates(keep='first', inplace=True)
        if 'tri' in file:
            system = systems_class[0]
        elif 'npi' in file:
            system = systems_class[1]
        else:
            system = systems_class[2]
        df_aux[column_name] = system
        df = pd.concat([df, df_aux],
                        axis=0, ignore_index=True)
        del df_aux
    df.drop_duplicates(keep='first', inplace=True)

    return df