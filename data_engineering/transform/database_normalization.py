#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.transform.common import config

import os
import pandas as pd


dir_path = os.path.dirname(os.path.realpath(__file__))

def calling_transformed_files(path, csv_from_path=['npi', 'npri', 'tri']):
    '''
    Function to call and concatenate all PRTRs or calling other files
    '''

    df = pd.DataFrame()
    for file in csv_from_path:
        df_u = pd.read_csv(f'{path}/{file}.csv', dtype={'national_substance_id': object})
        df = pd.concat([df, df_u],
                        axis=0, ignore_index=True)
        del df_u
        os.remove(f'{path}/{file}.csv')

    return df


def database_normalization():
    '''
    Function that looks for normalazing database
    '''

    # Calling columns for using and their names
    db_normalization_path = f'{dir_path}/../../ancillary/database_tables.yaml'
    db_normalization = config(db_normalization_path)['table']

    # Calling PRTR systems
    prtr = calling_transformed_files(f'{dir_path}/output')
    country_to_prtr = {'USA': 'TRI', 'AUS': 'NPI', 'CAN': 'NPRI'}
    prtr['prtr_system'] = prtr.country.apply(lambda x: country_to_prtr[x])
    country_to_ics = {'USA': 'USA_NAICS', 'AUS': 'ANZSIC', 'CAN': 'CAN_NAICS'}
    prtr['industry_classification_system'] = prtr.country.apply(lambda x: country_to_ics[x])
    prtr.drop(columns=['cas_number', 'national_substance_name'], inplace=True)

    # Creating 'national_facility_and_sector_id'
    grouping = ['national_facility_id', 'national_sector_code', 'prtr_system']
    prtr['national_facility_and_sector_id'] = pd.Series(prtr.groupby(grouping).ngroup())

    # Calling sectors
    sector = calling_transformed_files(f'{dir_path}/output',
                                    csv_from_path=['national_to_generic_sector'])

    # Calling substances
    substance = calling_transformed_files(f'{dir_path}/output',
                                    csv_from_path=['national_to_generic_substance'])

    # Calling transfer classes
    t_class = calling_transformed_files(f'{dir_path}/../../ancillary',
                                    csv_from_path=['National_to_generic_transfer'])
    t_class = t_class[pd.notnull(t_class.generic_transfer_class_id)]
    t_class.generic_system_comment =\
        t_class.groupby('generic_transfer_class_id')\
            .generic_system_comment\
                .transform(lambda g: g.fillna(method='bfill'))

    # Merging dataframes
    mergings = [[sector, ['national_sector_code', 'industry_classification_system']],
                [t_class, ['national_transfer_class_name', 'prtr_system']],
                [substance, ['national_substance_id', 'prtr_system']]]
    del sector, substance, t_class
    for merging in mergings:
        prtr = pd.merge(prtr, merging[0], how='left', on=merging[1])
    del mergings


    # Creating 'national_sector_and_classification_system_id'
    grouping = ['national_sector_code', 'national_sector_name', 'industry_classification_system']
    prtr['national_sector_and_classification_system_id'] = pd.Series(prtr.groupby(grouping).ngroup())

    # Creating individual tables
    for table, cols in db_normalization.items():
        df_table = prtr[cols['cols']]
        df_table = df_table.drop_duplicates(keep='first').reset_index(drop=True)
        df_table.to_csv(f'{dir_path}/output/{table}.csv',
                        index=False, sep=',')
    
    

if __name__ == '__main__':
    
    database_normalization()