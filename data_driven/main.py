#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.data_preparation.main import data_preparation_pipeline
from data_driven.modeling.main import modeling_pipeline


import logging
import argparse
logging.basicConfig(level=logging.INFO)


def machine_learning_pipeline(args):
    '''
    Function for creating the machine learning pipeline
    '''

    logger = logging.getLogger(' Data-driven modeling')

    logger.info(' Starting data-driven modeling')

    # Calling the data preparation pipeline
    df_ml = data_preparation_pipeline(args)

    # Calling the modeling pipeline
    modeling_pipeline(df_ml)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
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
    parser.add_argument('--feature_selection',
                        help='Would you like to select features',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--balanced_splitting',
                        help='Would you like to split the dataset in a balanced fashion',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')

    args = parser.parse_args()

    machine_learning_pipeline(args)