#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.transform.common import config

import os
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
conversion_factor = {'tonnes': 10**3,
                     'kg': 1,
                     'grams': 10**-3}

def opening_file(filename):
    '''
    Function to open the NPRI files
    '''

    # Calling columns for using and their names
    columns_path = f'{dir_path}/../../ancillary/NPRI_columns_for_using.yaml'
    columns_for_using = config(columns_path)


    # Calling NPI data file
    extracted_npi_path = f'{dir_path}/../extract/output/NPRI_{filename}.csv'
    df = pd.read_csv(extracted_npi_path, header=None,
                     skiprows=[0], engine='python',
                     error_bad_lines=False,
                     usecols=columns_for_using[filename].keys(),
                     names=columns_for_using[filename].values())
    
    return df
    

def transforming_npri():
    '''
    Function to transform NPRI raw data into the structure for the
    generic database
    '''

    # Calling NPRI transfers file
    df_npri_transfers = opening_file('transfers')

    # Calling NPRI disposals file
    df_npri_disposals = opening_file('disposals')

    # Removing on-site disposals
    df_npri_disposals =\
        df_npri_disposals.loc[~df_npri_disposals.Group.str.contains('On-site')]
    df_npri_disposals.drop(columns=['Group'],
                           inplace=True)

    # Concatenating both files
    df_npri = pd.concat([df_npri_transfers, df_npri_disposals],
                        ignore_index=True, axis=0)
    del df_npri_transfers, df_npri_disposals

    # Dropping records having g TEQ(ET) units
    df_npri = df_npri.loc[~df_npri.Units.str.contains('TEQ')]

    # Converting to kg
    func = lambda quanity, units: quanity*conversion_factor[units]
    df_npri['transfer_amount_kg'] = df_npri.apply(lambda row: \
        func(row['transfer_amount_kg'],
             row['Units']), axis=1)
    df_npri.drop(columns=['Units'], inplace=True)
    


if __name__ == '__main__':
    
    transforming_npri()