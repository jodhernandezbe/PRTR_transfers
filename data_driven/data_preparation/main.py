#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.data_preparation.initial_preprocessing import initial_data_preprocessing
from data_driven.data_preparation.preprocessing import data_preprocessing

import logging
import argparse

logging.basicConfig(level=logging.INFO)


def data_preparation_pipeline(args):
    '''
    Function to run the data preparation pipeline
    '''

    logger = logging.getLogger(' Data-driven modeling --> Data preparation')

    # Preliminary data preprocessing
    logger.info(' Running preliminary data preprocessing')
    df_ml = initial_data_preprocessing(logger, args)

    # Preprocessing
    logger.info(' Running data preprocessing')
    data = data_preprocessing(df_ml, args, logger)

    return data 
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rdbms',
                        help='The Relational Database Management System (RDBMS) you would like to use',
                        choices=['mysql', 'postgresql'],
                        type=str,
                        default='mysql')
    parser.add_argument('--password',
                        help='The password for using the RDBMS',
                        required=False,
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
    parser.add_argument('--including_groups',
                        help='Would you like to include the chemical groups',
                        choices=['Yes', 'No'],
                        type=str,
                        default='Yes')
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
    parser.add_argument('--encoding',
                        help='How you want to encode the non-ordinal categorical data',
                        choices=['one-hot-encoding', 'target-encoding'],
                        type=str,
                        required=False,
                        default='one-hot-encoding')
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
    parser.add_argument('--balanced_dataset',
                        help='Would you like to balance the dataset',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--how_balance',
                        help='What method to balance the dataset you would like to use (see options)',
                        choices=['random_oversample', 'smote', 'adasyn', 'random_undersample', 'near_miss'],
                        type=str,
                        required=False,
                        default='random_oversample')
    parser.add_argument('--dimensionality_reduction',
                        help='Would you like to apply dimensionality reduction?',
                        choices=['False',  'True'],
                        type=str,
                        required=False,
                        default='False')
    parser.add_argument('--dimensionality_reduction_method',
                        help='What method for dimensionality reduction would you like to apply?. In this point, after encoding, we only apply feature transformation by PCA - Principal Component Analysis or feature selection by Univariate Feature Selection with mutual information metric or RFC - Random Forest Classifier',
                        choices=['pca', 'ufs', 'rfc'],
                        type=str,
                        required=False,
                        default='pca')
    parser.add_argument('--balanced_splitting',
                        help='Would you like to split the dataset in a balanced fashion',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--before_2005',
                        help='Would you like to include data reported before 2005?',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--intput_file',
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


    args = parser.parse_args()

    data_preparation_pipeline(args)