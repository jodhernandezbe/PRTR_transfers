#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
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
    intervals['Value'] = intervals['Value'].apply(lambda x: f'Interval {x}')
    intervals = intervals.set_index(pd.IntervalIndex.from_arrays(intervals['From'], intervals['To'], closed='left'))['Value'] 
    df['transfer_amount_kg'] = df['transfer_amount_kg'].map(intervals)
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


def data_preprocessing(df, args, logger):
    '''
    Function to apply further preprocessing to the dataset
    '''

    # Organazing transfers flow rates
    logger.info(' Organizing the transfer flow rates')
    df = transfer_flow_rates(df,
                flow_handling=args.flow_handling,
                number_of_intervals=args.number_of_intervals)