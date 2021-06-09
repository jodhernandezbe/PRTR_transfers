#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.transform.common import config, dq_score

import os
import pandas as pd


dir_path = os.path.dirname(os.path.realpath(__file__))


def getting_worst_score(string, dq_matrix):
    '''
    Function to get the digits in the "reliability_score" column
    and obtain the reliability score
    '''

    score = 0
    for c in string:
        if c.isdigit():
            d = dq_matrix[c]
            if d > score:
                score = d
    return score


def transforming_npi():
    '''
    Function to transform NPI raw data into the structure for the
    generic database
    '''

    # Calling columns for using and their names
    columns_path = f'{dir_path}/../../ancillary/NPI_columns_for_using.yaml'
    columns_for_using = config(columns_path)

    # Calling values for reliability score
    dq_matrix = dq_score('NPI')

    # Calling NPI transfers data file
    extracted_npi_path = f'{dir_path}/../extract/output/NPI_transfers.csv'
    df_npi = pd.read_csv(extracted_npi_path, header=None,
                        skiprows=[0],
                        usecols=columns_for_using['transfers'].keys(),
                        names=columns_for_using['transfers'].values())

    # Keeping only the off-site transfers
    df_npi = df_npi.loc[df_npi['national_transfer_class_name'].str.contains(r'^Off-site.*')]

    # Taking the last year as the reporting year
    df_npi['reporting_year'] = df_npi['reporting_year'].str.extract(r'[0-9]{4}/([0-9]{4})')
    df_npi['reporting_year'] = df_npi['reporting_year'].astype(int)

    # Giving the reliability scores for the off-site transfers reported by facilities
    df_npi['reliability_score'] = df_npi['reliability_score'].apply(lambda score: getting_worst_score(score, dq_matrix))

    # Calling NPI substances data file (only CAS numbers and program substance IDs)
    extracted_npi_path = f'{dir_path}/../extract/output/NPI_substances.csv'
    df_npi_substances = pd.read_csv(extracted_npi_path, header=None,
                                    skiprows=[0],
                                    usecols=columns_for_using['substances'].keys(),
                                    names=columns_for_using['substances'].values())
    
    # Merging files
    df_npi = pd.merge(df_npi, df_npi_substances, how='inner', on='national_substance_id')
    del df_npi_substances

    # Adding country column
    df_npi['country'] = 'AUS'

    # Saving the transformed data
    decimals = pd.Series([2, 0], index=['transfer_amount_kg', 'reliability_score'])
    df_npi[['transfer_amount_kg', 'reliability_score']] =\
        df_npi[['transfer_amount_kg', 'reliability_score']].round(decimals)
    df_npi.to_csv(f'{dir_path}/output/npi.csv',
                index=False, sep=',')


if __name__ == '__main__':
    
    transforming_npi()