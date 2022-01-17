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
import openpyxl


dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path
logging.basicConfig(level=logging.INFO)

agrs_list = ['before_2005', 'including_groups', 'grouping_type',
            'flow_handling', 'number_of_intervals', 'output_column',
            'outliers_removal', 'balanaced_split', 'dimensionality_reduction',
            'dimensionality_reduction_method', 'balanced_dataset', 'classification_type',
            'target_class']


def isnot_string(val):
    '''
    Function to verify if it is a string
    '''

    try:
        int(float(val))
        return True
    except:
        return False


def to_numeric(val):
    '''
    Function to convert string to numeric
    '''

    val = str(val)
    try:
        return int(val)
    except ValueError:
        return float(val)


def checking_boolean(val):
    '''
    Function to check boolean values
    '''

    if val == 'True':
        return True
    elif val == 'False':
        return False
    else:
        return val


def data_preparation_pipeline(args):
    '''
    Function to run the data preparation pipeline
    '''

    # Desired file paths
    classification_type = args.classification_type.replace(' ', '_')
    zip_path = f'{dir_path}/output/data/transformed/{classification_type}/{args.id}'

    logger = logging.getLogger(' Data-driven modeling --> Data preparation')

    if not os.path.isfile(f'{zip_path}.npz'):

        filepath = f'{dir_path}/output/data/raw/{classification_type}/initial_dataset_{args.id}.csv'    
        if not os.path.isfile(filepath):

            # Preliminary data preprocessing
            logger.info(f' Running preliminary data preprocessing for data preparation id {args.id}')

            df_ml = initial_data_preprocessing(logger, args)
            df_ml.to_csv(filepath, index=False, sep=',')

        else:
            df_ml = pd.read_csv(filepath)

        # Preprocessing
        logger.info(f' Running data preprocessing for data preparation id {args.id}')

        data = data_preprocessing(df_ml, args, logger)

        # Saving the data
        logger.info(f' Saving data preparation id {args.id}')
        np.savez_compressed(zip_path,
                X_train=data['X_train'],
                Y_train=data['Y_train'],
                X_test=data['X_test'],
                Y_test=data['Y_test'])

        # Deleting the files and folders not needed
        os.remove(filepath)
        del data

    else:

        logger.info(f' Data {args.id} already preprocessed')


def main(args):
    '''
    Function to execute the data preparation pipeline
    '''

    if args.input_file == 'No':

        args_dict = vars(args)
        args_dict.update({par: checking_boolean(val) for par, val in args_dict.items()})

        ## Calling the data preparation pipeline
        data_preparation_pipeline(args)

    else:

        ## Opening file for data preprocesing params
        input_file_path = f'{dir_path}/input/data_preparations.xlsx'
        input_parms = pd.read_excel(input_file_path,
                                sheet_name='classification',
                                header=None,
                                skiprows=[0, 1],
                                usecols=range(15))
        input_parms[1] = input_parms[1].astype(int)
        myworkbook = openpyxl.load_workbook(input_file_path)
        worksheet = myworkbook['classification']

        for i, vals in input_parms.iterrows():

            run = vals[0]

            if run == 'No':

                vals = input_parms.iloc[i]

                args_dict = vars(args)
                args_dict.update({par: int(vals[idx+2]) if isnot_string(str(vals[idx+2])) else (None if str(vals[idx+2]) == 'None' else vals[idx+2]) for 
                idx, par in enumerate(agrs_list)})
                args_dict.update({'id': vals[1],
                                'save_info': 'Yes'})

                # Calling the data preparation pipeline
                data_preparation_pipeline(args)

                # Saving
                worksheet[f'A{i + 3}'].value = 'Yes'
                myworkbook.save(input_file_path)

            else:

                continue

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Data preparation pipeline')
    parser.add_argument('--rdbms',
                        help='The Relational Database Management System (RDBMS) you would like to use',
                        choices=['mysql', 'postgresql'],
                        type=str,
                        default='mysql')
    parser.add_argument('--password',
                        help='The password for using the RDBMS',
                        type=str)
    parser.add_argument('--username',
                        help='The username for using the RDBMS',
                        type=str,
                        default='root')
    parser.add_argument('--host',
                        help='The computer hosting for the database',
                        type=str,
                        default='127.0.0.1')
    parser.add_argument('--port',
                        help='Port used by the database engine',
                        type=str,
                        default='3306')
    parser.add_argument('--db_name',
                        help='Database name',
                        type=str,
                        default='PRTR_transfers')
    parser.add_argument('--id',
                        help='What id whould your like to use for the data preparation workflow',
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument('--before_2005',
                        help='Would you like to include data reported before 2005?',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--including_groups',
                        help='Would you like to include the chemical groups',
                        choices=['True', 'False'],
                        type=str,
                        default='True')
    parser.add_argument('--grouping_type',
                        help='How you want to calculate descriptors for the chemical groups',
                        choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        type=int,
                        required=False,
                        default=1)
    parser.add_argument('--flow_handling',
                        help='How you want to handle the transfer flow rates',
                        choices=[1, 2, 3, 4],
                        type=int,
                        required=False,
                        default=1)
    parser.add_argument('--number_of_intervals',
                        help='How many intervals would you like to use for the transfer flow rates',
                        type=int,
                        required=False,
                        default=10)
    parser.add_argument('--output_column',
                        help='What column would you like to keep as the classifier output',
                        choices=['generic', 'wm_hierarchy'],
                        type=str,
                        required=False,
                        default='generic')
    parser.add_argument('--outliers_removal',
                        help='Would you like to keep the outliers',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--balanaced_split',
                        help='Would you like to obtain an stratified train-test split?',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--dimensionality_reduction',
                        help='Would you like to apply dimensionality reduction?',
                        choices=['False',  'True'],
                        type=str,
                        required=False,
                        default='False')
    parser.add_argument('--dimensionality_reduction_method',
                        help='What method for dimensionality reduction would you like to apply?. In this point, after encoding, we only apply feature transformation by FAMD - Factor Analysis of Mixed Data or feature selection by UFS - Univariate Feature Selection with mutual infomation (filter method) or RFC - Random Forest Classifier (embedded method via feature importance)',
                        choices=['FAMD', 'UFS', 'RFC'],
                        type=str,
                        required=False,
                        default='FAMD')
    parser.add_argument('--balanced_dataset',
                        help='Would you like to balance the dataset',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--classification_type',
                        help='What kind of classification problem would you like ',
                        choices=['multi-model binary classification', 'multi-label classification', 'multi-class classification'],
                        type=str,
                        required=False,
                        default='multi-class classification')
    parser.add_argument('--target_class',
                        help='If applied, What is the target class (only for multi-model binary classification)',
                        choices=['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'Disposal', 'Sewerage', 'Treatment', 'Energy recovery', 'Recycling'],
                        type=str,
                        required=False,
                        default='M1')
    parser.add_argument('--input_file',
                        help='Do you have an input file?',
                        choices=['Yes', 'No'],
                        type=str,
                        required=False,
                        default='No')
    parser.add_argument('--save_info',
                        help='Would you like to save information?',
                        choices=['Yes', 'No'],
                        type=str,
                        required=False,
                        default='No')
    parser.add_argument('--data_fraction_to_use',
                        help='What fraction of the data would you like to use?',
                        type=float,
                        required=False,
                        default=1.0)

    args = parser.parse_args()
    main(args)




