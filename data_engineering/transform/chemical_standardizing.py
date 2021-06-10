#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.extract.srs_scraper import get_generic_name_by_cas
from data_engineering.transform.common import opening_files_for_sectors

import os
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))

def normalizing_chemicals():
    '''
    Function to normalize chemicals across programs
    '''

    # Calling file for crosswalking chemicals
    path_file = f'{dir_path}/../../ancillary/National_to_generic_chemicals.csv'
    df_cross = pd.read_csv(path_file, dtype={'national_substance_id': object})
    df_cross.drop(columns=['national_substance_id'], inplace=True)

    # Searching for PRTR files
    df_chem = opening_files_for_sectors()

    # Merging information to substances not having CAS or otherwise
    df_chem = pd.merge(df_chem, df_cross, on=['national_substance_name', 'note'], how='left')
    df_chem = df_chem.where(pd.notnull(df_chem), None)
    df_chem.drop_duplicates(keep='first', inplace=True)
    df_chem.reset_index(drop=True, inplace=True)

    # Replacing CAS numbers
    idx = df_chem[pd.notnull(df_chem.generic_substance_name)].index.tolist()
    df_chem.cas_number_x.iloc[idx] = df_chem.cas_number_y.iloc[idx]
    df_chem.drop(columns=['cas_number_y'], inplace=True)
    df_chem.rename(columns={'cas_number_x': 'cas_number'}, inplace=True)

    # Replacing generic substance id
    idx = df_chem[pd.notnull(df_chem.cas_number)].index.tolist()
    df_chem.generic_substance_id.iloc[idx] = df_chem.cas_number.iloc[idx].str.replace('-', '')
    
    # Looking for generic names
    df_chem.generic_substance_name = df_chem.apply(lambda row: row['generic_substance_name']
                                    if row['generic_substance_name']
                                    else get_generic_name_by_cas(casNum=row['cas_number']),
                                    axis=1)

    # Saving the transformed data
    df_chem.to_csv(f'{dir_path}/output/generic_substance.csv', sep=',', index=False)


if __name__ == '__main__':
    
    normalizing_chemicals()