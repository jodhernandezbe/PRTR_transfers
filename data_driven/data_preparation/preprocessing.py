#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from functools import partial
import pandas as pd
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path

def obtaining_intervals(df, vals_for_intervals, number_of_intervals, flow_handling, save_info, id):
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
    if save_info == 'Yes':
        intervals.to_csv(f'{dir_path}/output/intervals_for_flow_rates_for_params_id_{id}.csv',
                        index=False)

    return df


def transfer_flow_rates(df, id, flow_handling=1, number_of_intervals=10, save_info='No'):
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
        df = obtaining_intervals(df, quantile_values, number_of_intervals,
                            flow_handling, save_info, id)
    elif flow_handling == 4:
        df['transfer_amount_kg'] = df['transfer_amount_kg'].astype(int)
        max_value = df['transfer_amount_kg'].max()
        linear = np.linspace(start=0,
                            stop=max_value+2,
                            num=number_of_intervals+1,
                            dtype=int).tolist()
        df = obtaining_intervals(df, linear, number_of_intervals,
                flow_handling, save_info, id)
        
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


def categorical_data_encoding(df, cat_cols, id,
                            encoding='one-hot-encoding',
                            output_column='generic_transfer_class_id',
                            save_info='No'):
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
                                                    sparse=False)],
                                axis=1)
            else:
                df['sector'] = calc_smooth_mean(df1=df, df2=None,
                                                cat_name='generic_sector_code',
                                                target=f'{output_column}_label',
                                                weight=5)
            if save_info == 'Yes':
                df[[col for col in df.columns if 'sector' in col]].drop_duplicates(keep='first').to_csv(f'{dir_path}/output/generic_sector_for_params_id_{id}.csv',
                        index=False)
            df.drop(columns=['generic_sector_code'],
                    inplace=True)
        else:
            labelencoder = LabelEncoder()
            df[f'{col}_label'] = labelencoder.fit_transform(df[col])
            if save_info == 'Yes':
                df[[col, f'{col}_label']].drop_duplicates(keep='first').to_csv(f'{dir_path}/output/{col}_labelencoder_for_params_id_{id}.csv',
                        index=False)
            df.drop(columns=col, inplace=True)

    return df

def balancing_dataset(X, Y, how_balance):
    '''
    Function to balance the dataset based on the output clasess
    '''

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

def dimensionality_reduction(X_train, Y_train, dimensionality_reduction_method, X_test, feature_cols):
    '''
    Function to apply dimensionality reduction
    '''

    if dimensionality_reduction_method == 'PCA':
        # Select components based on the threshold for the explained variance
        pca = PCA()
        pca.fit(X_train)
        sum_explained_variance = 0
        threshold = 0.85
        for components_idx, variance in enumerate(pca.explained_variance_ratio_):
            sum_explained_variance += variance
            if sum_explained_variance >= threshold:
                break
        X_train_reduced = pca.transform(X_train)[:, 0: components_idx + 1]
        X_test_reduced = pca.transform(X_test)[:, 0: components_idx + 1]
        feature_names = None
    else:

        # Separating flows and sectors from chemical descriptors
        position_i = [i for i, val in enumerate(feature_cols) if ('transfer' not in val) and ('sector' not in val)]
        descriptors = [val for val in feature_cols if ('transfer' not in val) and ('sector' not in val)]
        feature_cols = [val for val in feature_cols if ('transfer' in val) or ('sector' in val)]
        X_train_d = X_train[:, position_i]
        X_test_d = X_test[:, position_i]
        X_train = np.delete(X_train, position_i, axis=1)
        X_test = np.delete(X_test, position_i, axis=1)

        # Removing columns that have constant values across the entire dataset
        constant_filter = VarianceThreshold(threshold=0)
        constant_filter.fit(X_train_d)
        X_train_reduced = constant_filter.transform(X_train_d)
        X_test_reduced = constant_filter.transform(X_test_d)
        descriptors = [descriptors[idx] for idx, val in enumerate(constant_filter.get_support()) if val]
        del X_train_d, X_test_d

        # Removing columns that have quasi-constant values across the entire dataset
        qconstant_filter = VarianceThreshold(threshold=0.01)
        qconstant_filter.fit(X_train_reduced)
        X_train_reduced = qconstant_filter.transform(X_train_reduced)
        X_test_reduced = qconstant_filter.transform(X_test_reduced)
        descriptors = [descriptors[idx] for idx, val in enumerate(qconstant_filter.get_support()) if val]

        if dimensionality_reduction_method == 'UFS':
            # Select half of the features
            n_features = X_train_reduced.shape[1] // 2
            skb = SelectKBest(partial(mutual_info_classif, random_state=0), k=n_features)
            skb.fit(X_train_reduced, Y_train)
            X_train_reduced = skb.transform(X_train_reduced)
            X_test_reduced = skb.transform(X_test_reduced)
            descriptors = [descriptors[idx] for idx, val in enumerate(skb.get_support()) if val]
        elif dimensionality_reduction_method == 'RFC':
            sel = SelectFromModel(RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=4))
            sel.fit(X_train_reduced, Y_train)
            X_train_reduced = sel.transform(X_train_reduced)
            X_test_reduced = sel.transform(X_test_reduced)
            descriptors = [descriptors[idx] for idx, val in enumerate(sel.get_support()) if val]

        # Concatenating
        X_train_reduced = np.concatenate((X_train, X_train_reduced), axis=1)
        X_test_reduced = np.concatenate((X_test, X_test_reduced), axis=1)

    return X_train_reduced, X_test_reduced


def data_preprocessing(df, args, logger):
    '''
    Function to apply further preprocessing to the dataset
    '''

    df = df.sample(100000)

    # Data before 2005 or not (green chemistry and engineering boom!)
    if args.before_2005 == 'True':
        pass
    else:
        df = df[df.reporting_year >= 2005]

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

    # Organazing transfers flow rates
    logger.info(' Organizing the transfer flow rates')
    df = transfer_flow_rates(df, args.id,
                flow_handling=args.flow_handling,
                number_of_intervals=args.number_of_intervals,
                save_info=args.save_info)

    # Dropping columns are not needed
    logger.info(' Dropping not needed columns')
    to_drop = ['transfer_record_id',
                'reporting_year',
                'reliability_score',
                'generic_substance_id',
                'national_facility_and_generic_sector_id']
    df.drop(columns=to_drop, inplace=True)
    num_cols = list(set(num_cols) - set(to_drop))
    
    # Organizing categorical data
    logger.info(' Encoding categorical features')
    df = categorical_data_encoding(df, cat_cols, args.id,
                            encoding=args.encoding,
                            output_column=col_to_keep,
                            save_info=args.save_info)

    # Dropping duplicates
    logger.info(' Dropping duplicated examples')
    df.drop_duplicates(keep='first', inplace=True)

    # Reducing the memory consumed by the data
    logger.info(' Donwcasting the dataset for saving memory')
    fcols = df.select_dtypes('float').columns
    icols = df.select_dtypes('integer').columns
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')
    del fcols, icols

    # Outliers detection
    if args.outliers_removal == 'True':
        logger.info(' Removing outliers')
        iso = IsolationForest(max_samples=100,
                            random_state=0,
                            contamination=0.2,
                            n_jobs=4)
        filter = iso.fit_predict(df[[col for col in df.columns if col not in col_to_keep]].values)
        df = df[filter == 1]
    else:
        pass

    # X and Y
    feature_cols = [col for col in df.columns if (col != f'{col_to_keep}_label')]
    X = df[feature_cols].values
    Y = df[f'{col_to_keep}_label']
    del df

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

    # Scaling
    logger.info(' Performing min-max scaling')
    scalerMinMax = MinMaxScaler()
    scalerMinMax.fit(X_train)
    X_train = scalerMinMax.transform(X_train)
    X_test = scalerMinMax.transform(X_test)

    # Balancing the dataset
    if args.balanced_dataset == 'True':
        logger.info(' Balancing the dataset')
        X_train, Y_train = balancing_dataset(X_train, Y_train, args.how_balance)
    else:
        pass

    # Dimensionality reduction
    if args.dimensionality_reduction == 'False':
        pass
    else:
        logger.info(f' Reducing dimensionality by {args.dimensionality_reduction_method.upper()}')
        X_train, X_test = dimensionality_reduction(X_train,
                                                Y_train,
                                                args.dimensionality_reduction_method,
                                                X_test,
                                                feature_cols)


    return {'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test}
