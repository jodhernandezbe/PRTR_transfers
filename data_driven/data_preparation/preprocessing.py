#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.data_preparation.mlsmote import get_minority_instace, MLSMOTE

from skmultilearn.model_selection import iterative_train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from prince import FAMD
import pandas as pd
import numpy as np
import os
import pickle
from functools import partial

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path


def industry_sector_encoding(
                            X_train, X_test,
                            id,
                            feature_cols,
                            feature_dtype,
                            flow_handling,
                            number_of_intervals,
                            save_info='No'
                            ):
    '''
    Function to encode the industry sector column
    '''

    n_train_idx = X_train.shape[0]
    df = pd.concat([pd.DataFrame(data=X_train, columns=feature_cols),
                    pd.DataFrame(data=X_test, columns=feature_cols)],
            axis=0, ignore_index=True)
    del X_train, X_test
    for c, t in feature_dtype.items():
        if c == 'transfer_amount_kg':
            df[c] = df[c].astype(int)
        else:
            df[c] = df[c].astype(t)
    if flow_handling in [3, 4]:
        df['transfer_amount_kg'] = df['transfer_amount_kg'].astype(int) / number_of_intervals
    df = pd.concat([df, pd.get_dummies(df.generic_sector_code,
                                        prefix='sector',
                                        sparse=False)],
                    axis=1)
    df.drop(columns=['generic_sector_code'],
        inplace=True)
    feature_cols_encoding = df.columns.tolist()
    X_train, X_test = df.iloc[0:n_train_idx].values, df.iloc[n_train_idx:].values

    if save_info == 'Yes':
        df[[col for col in df.columns if 'sector' in col]].drop_duplicates(keep='first').to_csv(f'{dir_path}/output/input_features/generic_sector_for_params_id_{id}.csv',
                index=False)

    del df

    return X_train, X_test, feature_cols_encoding


def dimensionality_reduction(X_train, Y_train, dimensionality_reduction_method, X_test,
                            feature_cols, feature_dtype, save_info, id, classification_type,
                            feature_cols_encoding=None):
    '''
    Function to apply dimensionality reduction
    '''

    if dimensionality_reduction_method == 'FAMD':

        # Select components based on the threshold for the explained variance
        X_train = pd.DataFrame(X_train, columns=feature_cols)
        X_test = pd.DataFrame(X_test, columns=feature_cols)
        for c, t in feature_dtype.items():
            X_train[c] = X_train[c].astype(t)
            X_test[c] = X_test[c].astype(t)

        famd = FAMD(
                    n_components=X_train.shape[1], n_iter=3, copy=True,
                    check_input=True, engine='auto', random_state=42
                    )
        famd.fit(X_train)
        sum_explained_variance = 0
        threshold = 0.95
        for components_idx, variance in enumerate(famd.explained_inertia_):
            sum_explained_variance += variance
            if sum_explained_variance >= threshold:
                break

        print(f'{components_idx+1} components explain {round(sum_explained_variance, 2)} of the variance for the data preprocessing {id}')
        X_train_reduced = famd.transform(X_train).values[:, 0:components_idx+1]
        X_test_reduced = famd.transform(X_test).values[:, 0:components_idx+1]

        if save_info == 'Yes':
            pickle.dump(famd, open(f'{dir_path}/output/transformation_models/famd_id_{id}.pkl', 'wb'))
            pd.Series(feature_cols).to_csv(f'{dir_path}/output/input_features/input_features_id_{id}.csv')
            pd.Series(feature_dtype).to_csv(f'{dir_path}/output/input_features/input_features_dtype_{id}.csv')
    
    else:

        # Separating flows and sectors from chemical descriptors
        position_i = [i for i, val in enumerate(feature_cols_encoding) if ('transfer' not in val) and ('sector' not in val) and ('epsi' not in val) and ('gva' not in val) and ('price_usd_g' not in val)]
        descriptors = [val for val in feature_cols_encoding if ('transfer' not in val) and ('sector' not in val) and ('epsi' not in val) and ('gva' not in val) and ('price_usd_g' not in val)]
        feature_cols_encoding = [val for val in feature_cols_encoding if ('transfer' in val) or ('sector' in val) or ('epsi' in val) and ('gva' in val) and ('price_usd_g' in val)]
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

        # Removing highly correlated variables
        cor_matrix = pd.DataFrame(X_train_reduced).corr().abs()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] >= 0.7)]
        X_train_reduced = pd.DataFrame(X_train_reduced).drop(to_drop, axis=1).values
        X_test_reduced = pd.DataFrame(X_test_reduced).drop(to_drop, axis=1).values
        descriptors = [val for idx, val in enumerate(descriptors) if not idx in to_drop]

        if dimensionality_reduction_method == 'UFS':
            # Select half of the features
            n_features = X_train_reduced.shape[1] // 3
            sel = SelectKBest(partial(mutual_info_classif, random_state=0), k=n_features)
        elif dimensionality_reduction_method == 'RFC':
            sel = SelectFromModel(RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1))
            
        selected_features = []
        if classification_type == 'multi-label classification':
            for label in range(Y_train.shape[1]):
                sel.fit(X_train_reduced, Y_train[:, label])
                selected_features.append(list(sel.get_support()))
        else:
            sel.fit(X_train_reduced, Y_train)
            selected_features.append(list(sel.get_support()))
        results = np.array(selected_features).any(axis=0)

        X_train_reduced = X_train_reduced[:, results]
        X_test_reduced = X_test_reduced[:, results]
        descriptors = [descriptors[idx] for idx, val in enumerate(results) if val]

        feature_cols = feature_cols + descriptors
        feature_dtype = {f: feature_dtype[f] for f in feature_cols}
        if save_info == 'Yes':
            pd.Series(feature_cols).to_csv(f'{dir_path}/output/input_features/input_features_id_{id}.csv', header=False)
            pd.Series(feature_dtype).to_csv(f'{dir_path}/output/input_features/input_features_dtype_{id}.csv')

        # Concatenating
        X_train_reduced = np.concatenate((X_train, X_train_reduced), axis=1)
        X_test_reduced = np.concatenate((X_test, X_test_reduced), axis=1)

    return X_train_reduced, X_test_reduced, feature_cols, feature_dtype


def balancing_dataset(X, Y, data_augmentation_algorithem):
    '''
    Function to balance the dataset based on the output clasess
    '''

    if data_augmentation_algorithem == 'MLSMOTE':

        X = pd.DataFrame(data=X)
        Y = pd.DataFrame(data=Y)

        X_sub, Y_sub, X, Y = get_minority_instace(X, Y)
        fraction = 0.5
        n_samples = int((fraction*(X.shape[0]+ X_sub.shape[0]) - X_sub.shape[0])/(1 - fraction))
        X_res, Y_res = MLSMOTE(X_sub, Y_sub, n_samples)
        del X_sub, Y_sub

        X_balanced = pd.concat([X, X_res], axis=0, ignore_index=True).values
        Y_balanced = pd.concat([Y, Y_res], axis=0, ignore_index=True).values

    elif data_augmentation_algorithem == 'SMOTE':

        smote = SMOTE(random_state=42, n_jobs=-1)
        X_balanced, Y_balanced = smote.fit_resample(X, Y)

    else:

        tl = TomekLinks()
        X_balanced, Y_balanced = tl.fit_resample(X, Y)


    return X_balanced, Y_balanced


def splitting_the_dataset(df, feature_cols, target_colum, classification_type, test_size=0.2, balanaced_split=True):
    '''
    Function to split the dataset
    '''

    X = df[feature_cols].values
    Y = df[target_colum].values
    if balanaced_split:
        if classification_type == 'multi-label classification':
            Y = np.array([[int(element) for element in row.split(' ')] for row in Y])
            X_train, Y_train, X_test, Y_test = iterative_train_test_split(X, Y, test_size=test_size)
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=test_size,
                                                    random_state=0,
                                                    shuffle=True,
                                                    stratify=Y)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size=test_size,
                                                    random_state=0,
                                                    shuffle=True)
        if classification_type == 'multi-label classification':
            Y_test = np.array([[int(element) for element in row.split(' ')] for row in Y_test])
            Y_train = np.array([[int(element) for element in row.split(' ')] for row in Y_train])
    del X, Y, df

    return X_train, X_test, Y_train, Y_test


def outlier_detection(logger, outliers_removal, X_train, X_test, Y_train, Y_test, feature_cols, feature_dtype):
    '''
    Isolation forest for anomaly detection
    '''

    until_train_index = X_train.shape[0]

    if outliers_removal:
        logger.info(' Removing outliers by Isolation Forest')
        rs = np.random.RandomState(0)
        outlier_detection = IsolationForest(max_samples=100, random_state=rs, contamination=0.2) 
        df = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], axis=0, ignore_index=True)
        df.columns = feature_cols
        for c, t in feature_dtype.items():
            df[c] = df[c].astype(t)
            if t == 'object': 
                df = pd.concat([df, pd.get_dummies(df[c],
                                            sparse=False)],
                        axis=1)
                df.drop(columns=[c],
                    inplace=True)
        outlier_detection.fit(df.values)
        filter = outlier_detection.predict(df.values)
        filter_train, filter_test = filter[0: until_train_index], filter[until_train_index:]
        X_train = X_train[filter_train==1]
        Y_train = Y_train[filter_train==1]
        X_test = X_test[filter_test==1]
        Y_test = Y_test[filter_test==1]
        
        return X_train, X_test, Y_train, Y_test
    else:
        return X_train, X_test, Y_train, Y_test



def data_preprocessing(df, args, logger):
    '''
    Function to apply further preprocessing to the dataset
    '''

    # Converting generic_sector_code from integer to string
    df['generic_sector_code'] = df['generic_sector_code'].astype(str)

    # Identifying numerical data and target column
    if args.output_column == 'generic':
        target_colum = 'generic_transfer_class_id'
    else:
        target_colum = 'transfer_class_wm_hierarchy_name'
    num_cols = df._get_numeric_data().columns
    if args.flow_handling in [3, 4]:
        num_cols = list(set(num_cols) - set(['transfer_amount_kg']))
        if args.dimensionality_reduction_method == 'FAMD':
            df['transfer_amount_kg'] = df['transfer_amount_kg'].astype(str)
    feature_cols = [col for col in df.columns if (col != target_colum)]
    feature_dtype = df[feature_cols].dtypes.apply(lambda x: x.name).to_dict()

    # Only for multi-model binary classification
    if args.target_class:
        df[target_colum] = df[target_colum].apply(lambda x: 1 if x == args.target_class else 0)
    else:
        if args.classification_type == 'multi-class classification':
            if target_colum == 'generic_transfer_class_id':
                existing_classes = {'M1': 0, 'M2': 1, 'M3': 2, 'M4': 3, 'M5':4 , 'M6': 5, 'M7': 6, 'M8': 7, 'M9': 8, 'M10': 9}
            else:
                existing_classes = {'Disposal': 0, 'Sewerage': 1, 'Treatment': 2, 'Energy recovery': 3, 'Recycling': 4}
            df[target_colum] = df[target_colum].apply(lambda x: existing_classes[x])

    # Dropping duplicates
    logger.info(' Dropping duplicated examples')
    df.drop_duplicates(keep='first', inplace=True)

    # Splitting the data
    logger.info(' Splitting the dataset')
    X_train, X_test, Y_train, Y_test = splitting_the_dataset(df, feature_cols, target_colum,
                                                        args.classification_type, test_size=0.2, 
                                                        balanaced_split=args.balanaced_split)

    # Scaling
    logger.info(' Performing min-max scaling')
    scalerMinMax = MinMaxScaler()
    num_idx = [i for i, col in enumerate(feature_cols) if col in num_cols]
    scalerMinMax.fit(X_train[:, num_idx])
    X_train[:, num_idx] = scalerMinMax.transform(X_train[:, num_idx])
    X_test[:, num_idx] = scalerMinMax.transform(X_test[:, num_idx])
    if args.save_info == 'Yes':
        min_scale = scalerMinMax.data_min_
        max_scale = scalerMinMax.data_max_
        pd.DataFrame({'feature': num_cols,
                     'min': min_scale,
                     'max': max_scale}).to_csv(f'{dir_path}/output/input_features/input_features_scaling_id_{args.id}.csv',
                     index=False)

    # Removing outliers
    X_train, X_test, Y_train, Y_test = outlier_detection(logger, args.outliers_removal, X_train, X_test, Y_train, Y_test, feature_cols, feature_dtype)
    if (args.dimensionality_reduction_method == 'FAMD') and (args.dimensionality_reduction):

        # # Dimensionality reduction
        logger.info(f' Reducing dimensionality by {args.dimensionality_reduction_method.upper()}')
        X_train, X_test, _, _ = dimensionality_reduction(X_train,
                                                 Y_train,
                                                 'FAMD',
                                                 X_test,
                                                 feature_cols,
                                                 feature_dtype,
                                                 args.save_info,
                                                 args.id,
                                                 args.classification_type)
    
    else:

        # Organizing industry sector
        logger.info(' Encoding industry sector')
        X_train, X_test, feature_cols_encoding = industry_sector_encoding(X_train, X_test,
                                args.id,
                                feature_cols,
                                feature_dtype,
                                args.flow_handling,
                                args.number_of_intervals,
                                save_info=args.save_info)

        if args.dimensionality_reduction:
            # Dimensionality reduction
            logger.info(f' Reducing dimensionality by {args.dimensionality_reduction_method.upper()}')
            X_train, X_test, feature_cols, feature_dtype= dimensionality_reduction(X_train,
                                                    Y_train,
                                                    args.dimensionality_reduction_method,
                                                    X_test,
                                                    feature_cols,
                                                    feature_dtype,
                                                    args.save_info, args.id,
                                                     args.classification_type,
                                                    feature_cols_encoding=feature_cols_encoding)

    if args.dimensionality_reduction:
        # Removing potential duplicates after dimensionality reduction
        train = pd.concat([pd.DataFrame(X_train),  pd.DataFrame(Y_train)], axis=1, ignore_index=True)
        train.drop_duplicates(keep='first', inplace=True)
        X_train = train.iloc[:, 0:X_train.shape[1]].values
        Y_train = train.iloc[:, X_train.shape[1]:].values
        del train
        test = pd.concat([pd.DataFrame(X_test),  pd.DataFrame(Y_test)], axis=1, ignore_index=True)
        test.drop_duplicates(keep='first', inplace=True)
        X_test = test.iloc[:, 0:X_train.shape[1]].values
        Y_test = test.iloc[:, X_train.shape[1]:].values
        del test

        if args.dimensionality_reduction_method == 'FAMD':

            #Scaling after FAMD
            logger.info(' Performing min-max scaling for features after FAMD')
            scalerMinMax = MinMaxScaler()
            scalerMinMax.fit(X_train)
            X_train = scalerMinMax.transform(X_train)
            X_test = scalerMinMax.transform(X_test)
            if args.save_info == 'Yes':
                min_scale = scalerMinMax.data_min_
                max_scale = scalerMinMax.data_max_
                pd.DataFrame({'feature': list(range(X_train.shape[1])),
                            'min': min_scale,
                            'max': max_scale}).to_csv(f'{dir_path}/output/input_features/famd_input_features_scaling_id_{args.id}.csv',
                            index=False)

    # Balancing the dataset
    if args.balanced_dataset:
        if args.classification_type == 'multi-label classification':
            data_augmentation_algorithem = 'MLSMOTE'
        elif args.classification_type == 'multi-class classification':
            data_augmentation_algorithem = 'SMOTE'
        else:
            data_augmentation_algorithem = 'TomekLinks'
        logger.info(f' Balancing the dataset by {data_augmentation_algorithem}')
        X_train, Y_train = balancing_dataset(X_train, Y_train, data_augmentation_algorithem)
    else:
        pass

    if args.classification_type != 'multi-label classification':
       Y_train = Y_train.reshape((Y_train.shape[0], 1))
       Y_test = Y_test.reshape((Y_test.shape[0], 1))

    return {'X_train': X_train,
            'Y_train': Y_train,
            'X_test': X_test,
            'Y_test': Y_test}
