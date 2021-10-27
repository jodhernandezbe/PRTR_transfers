#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.model_selection import train_test_split
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

def balancing_dataset(X, Y, col_to_keep, how_balance):
    '''
    Function to balance the dataset based on the output clasess
    '''

    Y.value_counts().to_csv(f'{dir_path}/output/counts_by_output_class_for_{col_to_keep}.csv')

    if how_balance == 'random_oversample':
        sampler = RandomOverSampler(random_state=42)
    elif how_balance == 'smote':
        sampler = SMOTE(k_neighbors=2)
    elif how_balance == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif how_balance == 'random_undersample':
        sampler = RandomUnderSampler(random_state=42)
    elif how_balance == 'near_miss':
        sampler = NearMiss()

    X_balanced, Y_balanced = sampler.fit_resample(X, Y)

    return X_balanced, Y_balanced


def data_preprocessing(df, args, logger):
    '''
    Function to apply further preprocessing to the dataset
    '''

    df = df.sample(50000)

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

    # Dropping columns are not needed
    logger.info(' Dropping not needed columns')
    to_drop = ['transfer_record_id',
                'reporting_year',
                'reliability_score',
                'generic_substance_id',
                'national_facility_and_generic_sector_id']
    df.drop(columns=to_drop, inplace=True)
    num_cols = list(set(num_cols) - set(to_drop))

    # Dropping columns with a lot missing values
    logger.info(' Dropping columns with a lot of missing values (> 0.8)')
    to_drop = df.columns[pd.isnull(df).sum(axis=0)/df.shape[0] > 0.8].tolist()
    df.drop(columns=to_drop, inplace=True)
    
    # Missing values imputation
    logger.info(' Imputing missing values')
    df.fillna(df.median(), inplace=True)

    # Dropping duplicates
    logger.info(' Dropping duplicated examples')
    df.drop_duplicates(keep='first', inplace=True)

    # Dropping correlated columns
    logger.info(' Dropping highly correlated features (> 0.95)')
    cor_matrix = df[num_cols].corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    df.drop(columns=to_drop, inplace=True)
    
    # Outliers detection
    if args.outliers_removal == 'True':
        logger.info(' Removing outliers')
        iso = IsolationForest(max_samples=100,
                            random_state=0,
                            contamination=0.2) 
        df = df[iso.fit_predict(df[[col for col in df.columns if col not in col_to_keep]].values) == 1]
    else:
        pass

    # X and Y
    feature_cols = [col for col in df.columns if col != f'{col_to_keep}_label']
    X = df[feature_cols].values
    Y = df[f'{col_to_keep}_label']
    del df

    # Scaling
    logger.info(' Performing min-max scaling')
    scalerMinMax = MinMaxScaler()
    X = scalerMinMax.fit_transform(X)

    # Balancing the dataset
    if args.balanced_dataset == 'True':
        logger.info(' Balancing the dataset')
        X, Y = balancing_dataset(X, Y, col_to_keep, args.how_balance)
    else:
        pass

    # Feature selection
    if args.feature_selection == 'True':
        logger.info(' Selecting features')
    else:
        pass

    # Splitting the data
    logger.info(' Splitting the dataset')
    if args.balanced_splitting == 'True':
        Target = Y
    else:
        Target = None
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                test_size=0.3,
                                                random_state=0,
                                                stratify=Target)

