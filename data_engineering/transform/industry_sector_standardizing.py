#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.transform.common import opening_files_for_sectors

import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import os
import re


dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path
ancillary_path = f'{dir_path}/../../ancillary' # ancillary folder path

# Dictionary for cross-walking to ISIC
sic = {
        'USA_NAICS': {
                    'file': 'USA_2017_NAICS_to_ISIC_4',
                    'cols': [
                                '2017 NAICS',
                                '2017 NAICS TITLE',
                                'ISIC',
                                'ISIC TITLE'
                            ]
                  },
        'CAN_NAICS': {
                    'file': 'CAN_2017_NAICS_to_ISIC_4',
                    'cols': [
                                '2017 NAICS',
                                '2017 NAICS TITLE',
                                'ISIC',
                                'ISIC TITLE'
                            ]
                  },
        'ANZSIC': {
                    'file': '2006_ANZSIC_to_ISIC_4',
                    'cols': [
                                '2006 ANZSIC',
                                '2006 ANZSIC TITLE',
                                'ISIC',
                                'ISIC TITLE'
                    ]
                   }
      }


def overlapping_groups(df):
    '''
    Function for creating generic industry sector groups based on
    ISIC codes overlapping
    '''

    isic_codes = df['isic_code'].unique().tolist()

    isic_to_generic = {}
    in_isic_to_generic = {}
    group_c = 0

    for idx_1, isic_code_1 in enumerate(isic_codes):

        if isic_code_1 in isic_to_generic.keys():
            continue
        else:

            overlapping_1 = list(set(in_isic_to_generic.keys()) & set(isic_code_1.split(';')))
            if overlapping_1:
                group_1 = in_isic_to_generic[overlapping_1[0]]
            else:
                group_c += 1
                group_1 = group_c

            isic_codes_not = np.setdiff1d(isic_codes[idx_1 + 1:],
                                          list(isic_to_generic.keys()))

            for isic_code_2 in isic_codes_not:
                overlapping_2 = list(set(in_isic_to_generic.keys()) & set(isic_code_2.split(';')))
                if overlapping_2:
                    group_2 = in_isic_to_generic[overlapping_2[0]]
                    isic_to_generic.update({isic_code_2: str(group_2)})
                    in_isic_to_generic.update({code: str(group_2) for code in isic_code_2.split(';') if code not in in_isic_to_generic.keys()})
                else:
                    overlapping_3 = list(set(isic_code_1.split(';')) & set(isic_code_2.split(';')))
                    if overlapping_3:
                        isic_to_generic.update({isic_code_2: str(group_1)})
                        in_isic_to_generic.update({code: str(group_1) for code in isic_code_2.split(';') if code not in in_isic_to_generic.keys()})

            isic_to_generic.update({isic_code_1: str(group_1)})
            in_isic_to_generic.update({code: str(group_1) for code in isic_code_1.split(';') if code not in in_isic_to_generic.keys()})

    df_isic_to_generic = pd.DataFrame(
        {'isic_code': list(isic_to_generic.keys()),
         'generic_sector_code': list(isic_to_generic.values())}
         )

    return df_isic_to_generic


def normalizing_sectors():
    '''
    Function for standardizing the national industry classification systems
    '''

    # Searching for PRTR files
    df_sectors = opening_files_for_sectors(usecols=['national_sector_code'],
                                        dtype={'national_sector_code': int},
                                        systems_class=['USA_NAICS', 'ANZSIC', 'CAN_NAICS'])

    # Calling files for cross-walking industry sectors
    df_converter = pd.DataFrame()
    df_isic = pd.DataFrame()
    for system, att in sic.items():

        file_name = att['file']
        file_path = f'{ancillary_path}/{file_name}.csv'
        df = pd.read_csv(file_path, usecols=att['cols'],
                        dtype={col: 'object' for col in att['cols']})

        national_code = att['cols'][0]
        national_name = att['cols'][1]
        df[national_code] = df[national_code].apply(lambda val: re.sub(r"[^0-9]+", "", val).lstrip('0'))
        df['ISIC'] = df['ISIC'].apply(lambda val: re.sub(r"[^0-9]+", "", val).lstrip('0'))
        df[national_code] = pd.to_numeric(df[national_code])
        df = df.loc[pd.notnull(df[national_code])]
        df[national_code] = df[national_code].astype('int')
        df[national_name] = df[national_name].str.strip().str.capitalize()
        df['ISIC TITLE'] = df['ISIC TITLE'].str.strip().str.capitalize()
        df_isic_aux = df[['ISIC', 'ISIC TITLE']]
        df.drop(['ISIC TITLE'], inplace=True, axis=1)
        df_isic_aux.rename(columns={'ISIC': 'isic_code',
                                    'ISIC TITLE': 'isic_name'},
                           inplace=True)

        df.sort_values(by=[national_code, 'ISIC'],
                       inplace=True)
        func = {'ISIC': lambda x: ';'.join(x.tolist())}
        df = df.groupby([national_code, national_name],
                        as_index=False)['ISIC'].agg(func)

        df['note'] = system
        df.rename(columns={'ISIC': 'isic_code',
                            national_code: 'national_sector_code',
                            national_name: 'national_sector_name'},
                  inplace=True)
        df_converter = pd.concat([df_converter, df],
                                 ignore_index=True,
                                 sort=False,
                                 axis=0)
        df_isic = pd.concat([df_isic, df_isic_aux],
                             ignore_index=True,
                             sort=False,
                             axis=0)
        del df, df_isic_aux

    # Keeping only those national sectors reporting to the PRTR systems
    df_converter = pd.merge(df_converter, df_sectors,
                        on=['national_sector_code', 'note'], how='right')
    print(df_converter.info())
    del df_sectors

    df_isic.drop_duplicates(subset=['isic_code'], inplace=True, keep='first')
    df_isic_to_generic = overlapping_groups(df_converter)
    df_converter = pd.merge(df_converter,
                            df_isic_to_generic,
                            on='isic_code',
                            how='inner')
    del df_isic_to_generic

    df_converter['isic_code'] = df_converter['isic_code'].str.split(';')
    df_converter = pd.DataFrame({
                                 col:np.repeat(df_converter[col].values,
                                               df_converter['isic_code'].str.len())
                                 for col in df_converter.columns.drop('isic_code')}
                                )\
        .assign(**{'isic_code': np.concatenate(df_converter['isic_code'].values)})\
        [df_converter.columns]


    df_national_to_generic = df_converter[['national_sector_code',
                                           'national_sector_name',
                                           'generic_sector_code',
                                           'note']]
    df_national_to_generic.drop_duplicates(keep='first',
                                           inplace=True)
    df_national_to_generic.reset_index(inplace=True, drop=True)
    df_national_to_generic['national_generic_sector_id'] =\
        pd.Series(df_national_to_generic.index.tolist()) + 1

    df_isic_to_generic = df_converter[['isic_code',
                                       'generic_sector_code']]
    del df_converter
    df_isic_to_generic.drop_duplicates(keep='first',
                                       inplace=True)
    df_isic_to_generic = pd.merge(df_isic_to_generic, df_isic,
                                  on='isic_code', how='left')
    df_isic_to_generic.reset_index(inplace=True, drop=True)
    df_isic_to_generic['isic_generic_id'] =\
        pd.Series(df_isic_to_generic.index.tolist()) + 1


    df_isic_to_generic.to_csv(f'{dir_path}/output/isic_to_generic.csv',
                              index=False, sep=',')
    df_national_to_generic.to_csv(f'{dir_path}/output/national_to_generic.csv',
                                  index=False, sep=',')


if __name__ == '__main__':

    normalizing_sectors()
