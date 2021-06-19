#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.transform.common import config, dq_score
from data_engineering.transform.naics_normalization import normalizing_naics

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
    

def checking_cas_number(cas):
    '''
    Function to check the CAS number consistency
    '''

    cas_r = cas[::-1]
    cas_n = ''
    idx = 1
    for c in cas_r:
        if (idx == 2) or (idx == 5):
            if c == '-':
                cas_n += c
                idx += 1
        else:
            cas_n += c
            idx += 1
    cas_n = cas_n[::-1]
    
    return cas_n


def checking_naics_codes_across_years(df):
    '''
    Function to check zero naics for facilities
    '''

    naics_codes = df[['national_facility_id', 'national_sector_code', 'reporting_year']].copy()
    naics_codes.drop_duplicates(keep='first', inplace=True)
    df.drop(columns='national_sector_code', inplace=True)

    def across_years(x):
        result = naics_codes[(naics_codes.national_facility_id == x) & (naics_codes.national_sector_code != 0)]
        if not result.empty:
            return result.loc[result.reporting_year == result.reporting_year.max(), 'national_sector_code']
        else:
            return 0 

    naics_codes.national_sector_code = naics_codes.apply(lambda row: row['national_sector_code']
                    if row['national_sector_code'] != 0
                    else across_years(row['national_facility_id']), axis=1)

    df = pd.merge(df, naics_codes, on=['national_facility_id', 'reporting_year'], how='left')
    df.national_sector_code = df.national_sector_code.astype(int)

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
        df_npri_disposals.loc[~ (df_npri_disposals.Group.str.contains('On-site').astype(bool))]
    df_npri_disposals.drop(columns=['Group'],
                           inplace=True)

    # Concatenating both files
    df_npri = pd.concat([df_npri_transfers, df_npri_disposals],
                        ignore_index=True, axis=0)
    del df_npri_transfers, df_npri_disposals

    # Excluding some substances
    id_for_excluding = ['NA - 22', 'NA - M14']
    df_npri = df_npri[~(df_npri['national_substance_id'].isin(id_for_excluding))]

    # Dropping records having g TEQ(ET) units
    df_npri = df_npri.loc[~ (df_npri.Units.str.contains('TEQ').astype(bool))]

    # Converting to kg
    func = lambda quanity, units: quanity*conversion_factor[units]
    df_npri['transfer_amount_kg'] = df_npri.apply(lambda row: \
        func(row['transfer_amount_kg'],
             row['Units']), axis=1)
    df_npri.drop(columns=['Units'], inplace=True)

    # Calling values for reliability score
    dq_matrix = dq_score('NPRI')

    # Giving the reliability scores for the off-site transfers reported by facilities
    df_npri = df_npri.where(pd.notnull(df_npri), None)
    df_npri['reliability_score'] = df_npri['reliability_score'].apply(lambda s: dq_matrix[s] if s else 5)

    # Adding country column
    df_npri['country'] = 'CAN'

    # Adding CAS number column
    df_npri['cas_number'] = df_npri['national_substance_id'].apply(lambda x:
        checking_cas_number(x.strip().lstrip('0'))
        if 'NA' not in x else None)

    # Checking NAICS codes by facility
    df_npri = checking_naics_codes_across_years(df_npri)
    df_npri = df_npri[df_npri.national_sector_code != 0]

    # Crosswalking NAICS codes
    df_npri = normalizing_naics(df_npri, system='CAN')

    # Saving the transformed data
    decimals = pd.Series([2, 0], index=['transfer_amount_kg', 'reliability_score'])
    df_npri[['transfer_amount_kg', 'reliability_score']] =\
        df_npri[['transfer_amount_kg', 'reliability_score']].round(decimals)
    df_npri.to_csv(f'{dir_path}/output/npri.csv',
                   index=False, sep=',')


if __name__ == '__main__':
    
    transforming_npri()