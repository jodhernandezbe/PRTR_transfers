#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.rdkit_descriptors import information_for_set_of_chems
from data_engineering.extract.nlm_scraper import looking_for_structure_details as nlm
from data_engineering.extract.pubchem_scraper import looking_for_structure_details as pubchem
from data_driven.opening_dataset import opening_dataset

import logging
import argparse
import os
import random
import pandas as pd
from scipy.stats import zscore
logging.basicConfig(level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path


def chemical_group_descriptors(group, generic_substance_id, grouping_type=1):
    '''
    Function to calculate the descriptors for chemical groups
    
    Options:

    (1) mean value (default)
    (2) median value
    (3) min value
    (4) max value
    (5) random value
    (6) random chemical
    (7) keep chemicals (keep all chemicals having non-null records (95%))
    '''

    if grouping_type == 1:
        if group.shape[0] != 1:
            # Removing outliers
            z_scores = zscore(group)
            abs_z_scores = z_scores.abs()
            # Filling NaN zscore values
            abs_z_scores.fillna(0, inplace=True)
            filtered_entries = (abs_z_scores < 3).all(axis=1)
            group = group[filtered_entries]
        group = group.mean()
    elif grouping_type == 2:
        group = group.median()
    elif grouping_type == 3:
        group = group.min()
    elif grouping_type == 4:
        group = group.max()
    elif grouping_type == 5:
        group.reset_index(drop=True, inplace=True)
        group = pd.Series(group.apply(lambda x: pd.Series(random.choice(x)), axis=0).to_dict('records')[0])
    elif grouping_type == 6:
        group = pd.Series(group.sample(1, random_state=0).to_dict('records')[0])
    elif grouping_type == 7:
        group = group[group.count(axis=1)/group.shape[1] >= 0.95]
        group['generic_substance_id'] = generic_substance_id

    return group


def looking_for_smiles(cas_number):
    '''
    Function to look for the SMILES using both the NLM and PubChem databases
    '''
    smiles = pubchem(cas_number)
    if not smiles:
        smiles = nlm(cas_number)

    return smiles


def initial_data_preprocessing(logger, db_name,
                                grouping_type, including_groups):
    '''
    Function for a preliminary preprocessing of the data
    '''

    logger = logging.getLogger(' Data-driven modeling -> preliminary data preprocessing')

    cas_dict = {'chemical': 'chemical_in_category_cas',
                'substance': 'cas_number'}
    datasets = ['chemical', 'substance', 'record']

    df_chem = pd.DataFrame()

    for dataset in datasets:

        # Opening and/or creating the dataset
        logger.info(f' Fetching the needed information for the {dataset} dataset from the {db_name} database')

        if os.path.isfile(f'{dir_path}/output/{dataset}.csv'):
            df = pd.read_csv(f'{dir_path}/output/{dataset}.csv',
                            dtype={'generic_substance_id': object})
        else:
            df = opening_dataset(args, dataset)

            if (dataset in cas_dict.keys()):
                
                # Looking for SMILES
                logger.info(f' Looking for SMILES for compounds belonging to the {dataset}s list')
                df['smiles'] = df[cas_dict[dataset]].apply(lambda cas: looking_for_smiles(cas))

                # Looking for chemical descriptors
                df = df[pd.notnull(df['smiles'])]
                logger.info(f' Looking for descriptors for compounds belonging to the {dataset}s list')
                df = information_for_set_of_chems(cas_dict[dataset], df)
                df.reset_index(drop=True, inplace=True)
                df.drop(columns=['smiles'], inplace=True)

            # Saving information for further use and speeding up ML pipeline
            df.to_csv(f'{dir_path}/output/{dataset}.csv', index=False, sep=',')

        # Organizing descriptors for chemicals belonging to the groups
        if (dataset == 'chemical') and (including_groups == 'Yes'):
            df.drop(columns=['chemical_in_category_cas'], inplace=True)
            descriptors = [col for col in df.columns if col != 'generic_substance_id']
            df = df.groupby(['generic_substance_id'], as_index=False)\
                            .apply(lambda group: chemical_group_descriptors(
                                    group[descriptors],
                                    group.generic_substance_id.unique()[0],
                                    grouping_type=grouping_type)
                                    )
            df_chem = pd.concat([df_chem, df], ignore_index=True, axis=0)
        elif (dataset == 'substance'):
            df.drop(columns=['cas_number'], inplace=True)
            df_chem = pd.concat([df_chem, df], ignore_index=True, axis=0)
        elif (dataset == 'record'):
            df_ml = pd.merge(df, df_chem, on='generic_substance_id', how='inner')

        del df

    return df_ml


def data_driven_pipeline(args):
    '''
    Function to run the ML pipeline
    '''

    logger = logging.getLogger(' Data-driven modeling')

    logger.info(' Starting data-driven modeling')

    # Opening and preliminary data preprocessing
    df_ml = initial_data_preprocessing(logger,
                                    args.db_name,
                                    args.grouping_type,
                                    args.including_groups)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rdbms',
                        help='The Relational Database Management System (RDBMS) you would like to use',
                        choices=['MySQL', 'PostgreSQL'],
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
    parser.add_argument('--grouping_type',
                        help='How you want to calculate descriptors for the chemical groups',
                        choices=[1, 2, 3, 4, 5, 6, 7],
                        type=int,
                        required=False,
                        default=1)
    parser.add_argument('--including_groups',
                        help='Would you like to include the chemical groups',
                        choices=['Yes', 'No'],
                        type=str,
                        default='Yes')

    args = parser.parse_args()

    data_driven_pipeline(args)