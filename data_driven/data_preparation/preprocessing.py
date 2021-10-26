#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path

def obtaining_intervals(df, vals_for_intervals, number_of_intervals, flow_handling):
    '''
    Function to obtain the intervals for the flows
    '''

    num_different_elements = len(vals_for_intervals)
    if flow_handling == 3:
        vals_for_intervals[-1] = vals_for_intervals[-1] + 2
    intervals = pd.DataFrame({'From': vals_for_intervals[0:num_different_elements-1],
                                'To': vals_for_intervals[1:]})
    intervals['Value'] = pd.Series(intervals.index.tolist()) + 1
    intervals = intervals.set_index(pd.IntervalIndex.from_arrays(intervals['From'], intervals['To'], closed='left'))['Value'] 
    df['transfer_amount_kg'] = df['transfer_amount_kg'].map(intervals)
    df['transfer_amount_kg'] = df['transfer_amount_kg'].astype(object)
    # Saving equal-width intervals 
    intervals = intervals.reset_index()
    intervals.rename(columns={'index': 'Flow rate interval [kg]'}, inplace=True)
    if flow_handling == 3:
        string = 'balanced'
    else:
        string = 'equal-width'
    intervals.to_csv(f'{dir_path}/output/{number_of_intervals}_{string}_intervals_for_flow_rates.csv',
                    index=False)

    return df


def transfer_flow_rates(df, flow_handling=1, number_of_intervals=10):
    '''
    Function to organize the transfer flow rates

    Options:

    (1) Float values (default)
    (2) Integer values
    (3) m balanced intervals split by quantiles
    (4) m non-balanced equal-width intervals
    '''

    if flow_handling == 2:
        df['transfer_amount_kg'] = df['transfer_amount_kg'].astype(int)
    elif flow_handling == 3:
        df['transfer_amount_kg'] = df['transfer_amount_kg'].astype(int)
        quantiles = np.linspace(start=0, stop=1,
                                num=number_of_intervals+1)
        quantile_values = df['transfer_amount_kg'].quantile(quantiles).astype(int).unique().tolist()
        df = obtaining_intervals(df, quantile_values, number_of_intervals, flow_handling)
    elif flow_handling == 4:
        df['transfer_amount_kg'] = df['transfer_amount_kg'].astype(int)
        max_value = df['transfer_amount_kg'].max()
        linear = np.linspace(start=0,
                            stop=max_value+2,
                            num=number_of_intervals+1,
                            dtype=int).tolist()
        df = obtaining_intervals(df, linear, number_of_intervals, flow_handling)
        
    return df


def calc_smooth_mean(df1, df2, cat_name, target, weight):
    '''
    Function to apply target encoding

    Source: https://maxhalford.github.io/blog/target-encoding/
    '''
    
    # Compute the global mean
    mean = df1[target].mean()

    # Compute the number of values and the mean of each group
    agg = df1.groupby(cat_name)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + weight * mean) / (counts + weight)

    # Replace each value by the according smoothed mean
    if df2 is None:
        return df1[cat_name].map(smooth)
    else:
        return df1[cat_name].map(smooth),df2[cat_name].map(smooth.to_dict())


def categorical_data_encoding(df, cat_cols,
                            encoding='one-hot-encoding',
                            output_column='generic_transfer_class_id'):
    '''
    Function to encode the categorical features
    '''

    for col in cat_cols:
        if col == 'transfer_amount_kg':
            continue
        elif col == 'generic_sector_code':
            if encoding == 'one-hot-encoding':
                df = pd.concat([df, pd.get_dummies(df.generic_sector_code,
                                                    prefix='sector',
                                                    sparse=True)],
                                axis=1)
            else:
                df['sector'] = calc_smooth_mean(df1=df, df2=None,
                                                cat_name='generic_sector_code',
                                                target=f'{output_column}_label',
                                                weight=5)

            df[[col for col in df.columns if 'sector' in col]].drop_duplicates(keep='first').to_csv(f'{dir_path}/output/generic_sector_{encoding}.csv',
                    index=False)
            df.drop(columns=['generic_sector_code'],
                    inplace=True)
        else:
            labelencoder = LabelEncoder()
            df[f'{col}_label'] = labelencoder.fit_transform(df[col])

            df[[col, f'{col}_label']].drop_duplicates(keep='first').to_csv(f'{dir_path}/output/{col}_labelencoder.csv',
                    index=False)
            df.drop(columns=col, inplace=True)

    return df


def data_preprocessing(df, args, logger):
    '''
    Function to apply further preprocessing to the dataset
    '''

    # Organazing transfers flow rates
    logger.info(' Organizing the transfer flow rates')
    df = transfer_flow_rates(df,
                flow_handling=args.flow_handling,
                number_of_intervals=args.number_of_intervals)

    # Converting generic_sector_code from integer to string
    df['generic_sector_code'] = df['generic_sector_code'].astype(object)

    # Identifying categorical data
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols) - set(['generic_substance_id']))
    if args.output_column == 'generic':
        col_to_keep = 'generic_transfer_class_id'
    else:
        col_to_keep = 'transfer_class_wm_hierarchy_name'
    index = cat_cols.index(col_to_keep)
    first_element = cat_cols[0]
    cat_cols[0] = col_to_keep
    cat_cols[index] = first_element

    # Organizing categorical data
    logger.info(' Encoding categorical features')
    df = categorical_data_encoding(df, cat_cols,
                            encoding=args.encoding,
                            output_column=col_to_keep)

    # Grouping
    df.drop(columns=['transfer_record_id',
                    'reliability_score',
                    'generic_substance_id'],
                    inplace=True)
    grouping_cols = ['reporting_year',
                    'national_facility_and_generic_sector_id',
                    'transfer_amount_kg']
    [grouping_cols.append(col) for col in df.columns if ('sector' in col) and (col not in grouping_cols) and (col != f'{col_to_keep}_label')]
    df = df.groupby(grouping_cols, as_index=False).agg(lambda x: list(x))
    df.reset_index(inplace=True, drop=True)

    
    # print(df.shape)
    # print(df.info())
    # print(df.columns)
    # # Outliers detection
    # if args.outliers_removal == 'True':
    #     iso = IsolationForest(max_samples=100,
    #                     random_state=0,
    #                     contamination=0.2) 
    #     df = df[iso.fit_predict(df[[col for col in df.columns if col not in col_to_keep]].values) == 1]
    # else:
    #     pass

    # print(df.shape)

    # # Balancing the dataset

    # # Feature selection
