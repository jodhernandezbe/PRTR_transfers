# -*- coding: utf-8 -*-
# !/usr/bin/env python

# Importing libraries
import pandas as pd
import os
import re
#pd.options.mode.chained_assignment = None

dir_path = os.path.dirname(os.path.realpath(__file__))

def searching_equivalent_naics(row, df_naics):
    '''
    Function to search for the NAICS equivalent for years before 2017
    '''

    code = row['national_sector_code']
    # Selecting years before 2017 to merge
    if (df_naics['2017 NAICS Code'] == code).any():
        return code
    else:
        code_2017 = None
        for year in range(2012, 1996, -5):
            df_result = df_naics.loc[df_naics[f'{year} NAICS Code'] == code]
            df_result.reset_index(drop=True, inplace=True)
            if df_result.empty:
                continue
            else:
                code_2017 = df_result['2017 NAICS Code'].iloc[0]
        if code_2017:
            return code_2017
        else:
            return code


def normalizing_naics(df_system, system='USA'):
    '''
    Function to normalize NAICS codes
    '''

    regex = re.compile(f'{system}_\d{{4}}_to_(\d{{4}})_NAICS.csv')
    func = lambda name_file: int(re.search(regex, name_file).group(1))

    # Calling NAICS changes
    path_naics = f'{dir_path}/../../ancillary'
    naics_files = [file for file in os.listdir(path_naics)
                   if re.search(regex, file)]
    naics_files.sort(key=func, reverse=True)

    # Concatenating NAICS years
    for i, file in enumerate(naics_files):
        df_naics_aux = pd.read_csv(f'{path_naics}/{file}',
                             low_memory=False,
                             sep=',', header=0)
        if i == 0:
            df_naics = df_naics_aux
            df_naics.drop(columns=['2012 NAICS Title', '2017 NAICS Title'],
                        inplace=True)
        else:
            cols = df_naics_aux.iloc[:, :].columns.tolist()
            cols_for_merge = cols[2]
            df_naics = pd.merge(df_naics, df_naics_aux[cols[0:3:2]], how='outer', on=cols_for_merge)
            df_naics.drop_duplicates(keep='first', inplace=True)

    # crosswalking NAICS codes
    df_naics = df_naics.fillna(0)
    for col in df_naics.columns:
        df_naics[col] = df_naics[col].astype(int)

    df_system['national_sector_code'] = df_system.apply(lambda row: searching_equivalent_naics(row, df_naics),
                                                    axis=1)
    df_system['national_sector_code'] = df_system['national_sector_code'].astype(pd.Int32Dtype())
    
    return df_system