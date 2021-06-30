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
import pandas as pd
logging.basicConfig(level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path


def looking_for_smiles(cas_number):
    '''
    Function to look for the SMILES using both the NLM and PubChem databases
    '''
    smiles = pubchem(cas_number)
    if not smiles:
        smiles = nlm(cas_number)

    return smiles


def data_driven_pipeline(args):

    logger = logging.getLogger(' Data driven')

    logger.info(' Starting data driven')

    # Opening and/or creating the database
    logger.info(f' Fetching the needed information from the {args.db_name} database')
    dfs = opening_dataset(args)

    cas_dict = {'chemical': 'chemical_in_category_cas',
                'substance': 'cas_number'}
    for key, cas_col in cas_dict.items():
        df_chem = dfs[key]

        # Looking for SMILES
        logger.info(f' Looking for SMILES for compounds belonging to the {key} list')
        if os.path.isfile(f'{dir_path}/output/{key}.csv'):
            df_chem = pd.read_csv(f'{dir_path}/output/{key}.csv')
        else:
            df_chem['smiles'] = df_chem[cas_col].apply(lambda cas: looking_for_smiles(cas))
            df_chem.to_csv(f'{dir_path}/output/{key}.csv', index=False, sep=',')

        # Looking for chemical descriptors
        logger.info(f' Looking for descriptors for compounds belonging to the {key} list')
        df_chem = information_for_set_of_chems(cas_col, df_chem)

        print(df_chem.info())
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rdbms',
                        help='The Relational Database Management System (RDBMS) you would like to use',
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

    args = parser.parse_args()

    data_driven_pipeline(args)