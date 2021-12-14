#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import os
import pickle
from functools import partial
from sqlalchemy.sql.expression import column

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path


def calc_smooth_mean(df1, df2, cat_name, target, weight):
    '''
    Function to apply target encoding

    Based on https://maxhalford.github.io/blog/target-encoding/
    '''
    
    # Compute the global mean
    le = LabelEncoder()
    le.fit(df1[target])
    df1[f'{target}_label'] = le.transform(df1[target])
    mean = df1[f'{target}_label'].mean()

    # Compute the number of values and the mean of each group
    agg = df1.groupby(cat_name)[f'{target}_label'].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Droping
    df1.drop(columns=[f'{target}_label'], inplace=True)

    # Compute the "smoothed" means
    smooth = (counts * means + weight * mean) / (counts + weight)

    # Replace each value by the according smoothed mean
    if df2 is None:
        return df1[cat_name].map(smooth)
    else:
        return df1[cat_name].map(smooth),df2[cat_name].map(smooth.to_dict())


def industry_sector_encoding(df, id,
                            encoding='one-hot-encoding',
                            output_column='generic_transfer_class_id',
                            save_info='No'):
    '''
    Function to encode the industry sector column
    '''

    if encoding == 'one-hot-encoding':
        df = pd.concat([df, pd.get_dummies(df.generic_sector_code,
                                            prefix='sector',
                                            sparse=False)],
                        axis=1)
    else:
        df['sector'] = calc_smooth_mean(df1=df, df2=None,
                                        cat_name='generic_sector_code',
                                        target=output_column,
                                        weight=5)
    if save_info == 'Yes':
        df[[col for col in df.columns if 'sector' in col]].drop_duplicates(keep='first').to_csv(f'{dir_path}/output/generic_sector_for_params_id_{id}.csv',
                index=False)
    df.drop(columns=['generic_sector_code'],
            inplace=True)
       

    return df


def dimensionality_reduction(X_train, Y_train, dimensionality_reduction_method, X_test,
                            feature_cols, save_info, id):
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
        if save_info == 'Yes':
            pca = PCA(n_components=components_idx)
            pca.fit(X_train)
            pickle.dump(pca, open(f'{dir_path}/output/pca_{components_idx + 1}_components_id_{id}.pkl', 'wb'))
            pd.Series(feature_cols).to_csv(f'{dir_path}/output/input_features_id_{id}.csv')
    else:

        # Separating flows and sectors from chemical descriptors
        position_i = [i for i, val in enumerate(feature_cols) if ('transfer' not in val) and ('sector' not in val) and ('epsi' not in val)]
        descriptors = [val for val in feature_cols if ('transfer' not in val) and ('sector' not in val) and ('epsi' not in val)]
        feature_cols = [val for val in feature_cols if ('transfer' in val) or ('sector' in val) or ('epsi' in val)]
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
            n_features = X_train_reduced.shape[1] // 3
            skb = MultiOutputRegressor(SelectKBest(partial(mutual_info_regression, random_state=0), k=n_features))
            skb.fit(X_train_reduced, Y_train)
            results = np.array([list(estimator.get_support()) for estimator in skb.estimators_]).T.any(axis=1)
        elif dimensionality_reduction_method == 'RFR':
            sel = MultiOutputRegressor(SelectFromModel(RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)))
            sel.fit(X_train_reduced, Y_train)
            results = np.array([list(estimator.get_support()) for estimator in sel.estimators_]).T.any(axis=1)
        X_train_reduced = X_train_reduced[:, results]
        X_test_reduced = X_test_reduced[:, results]
        descriptors = [descriptors[idx] for idx, val in enumerate(results) if val]

        feature_cols = feature_cols + descriptors
        pd.Series(feature_cols).to_csv(f'{dir_path}/output/input_features_id_{id}.csv', header=False)

        # Concatenating
        X_train_reduced = np.concatenate((X_train, X_train_reduced), axis=1)
        X_test_reduced = np.concatenate((X_test, X_test_reduced), axis=1)

    return X_train_reduced, X_test_reduced, feature_cols


def data_preprocessing(df, args, logger):
    '''
    Function to apply further preprocessing to the dataset
    '''

    # Converting generic_sector_code from integer to string
    df['generic_sector_code'] = df['generic_sector_code'].astype(object)

    # Identifying numerical data and target column
    if args.output_column == 'generic':
        target_colum = 'generic_transfer_class_id'
    else:
        target_colum = 'transfer_class_wm_hierarchy_name'
    num_cols = df._get_numeric_data().columns
    if args.flow_handling in [3, 4]:
        num_cols = list(set(num_cols) - set(['transfer_amount_kg']))
    
    # Organizing industry sector
    logger.info(' Encoding industry sector')
    df = industry_sector_encoding(df, args.id,
                            encoding=args.encoding,
                            output_column=target_colum,
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
    if args.outliers_removal:
        logger.info(' Removing outliers')
        iso = IsolationForest(max_samples=100,
                            random_state=0,
                            contamination=0.2,
                            n_jobs=-1)
        filter = iso.fit_predict(df[[col for col in df.columns if col not in target_colum]].values)
        df = df[filter == 1]
    else:
        pass

    # Splitting the data
    feature_cols = [col for col in df.columns if (col != target_colum)]
    X = df[feature_cols].values
    Y = df[target_colum].values
    logger.info(' Splitting the dataset')
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                Y,
                                                test_size=0.2,
                                                random_state=0,
                                                shuffle=True)
    del X, Y, df
    Y_train
    Y_test

    # Balancing the dataset
    logger.info(' Applying random oversampling')



    # Scaling
    logger.info(' Performing min-max scaling')
    scalerMinMax = MinMaxScaler()
    scalerMinMax.fit(X_train)
    X_train = scalerMinMax.transform(X_train)
    X_test = scalerMinMax.transform(X_test)
    if args.save_info == 'Yes':
        min_scale = scalerMinMax.data_min_
        max_scale = scalerMinMax.data_max_
        pd.DataFrame({'feature': feature_cols,
                    'min': min_scale,
                    'max': max_scale}).to_csv(f'{dir_path}/output/scaling_id_{args.id}.csv',
                    index=False)

    # Dimensionality reduction
    if args.dimensionality_reduction:
        logger.info(f' Reducing dimensionality by {args.dimensionality_reduction_method.upper()}')
        X_train, X_test, feature_cols = dimensionality_reduction(X_train,
                                                Y_train,
                                                args.dimensionality_reduction_method,
                                                X_test,
                                                feature_cols,
                                                args.save_info, args.id)
    else:
        pass


    return {'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test,
            'feature_cols': feature_cols,
            'num_cols': [col for col in num_cols if col in feature_cols]}
