#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import warnings
warnings.filterwarnings("ignore")

from data_driven.data_preparation.main import data_preparation_pipeline
from data_driven.modeling.main import modeling_pipeline
from data_driven.modeling.evaluation import data_driven_models_ranking, prediction_evaluation
from data_driven.modeling.tuning import parameter_tuning

import time
import openpyxl
import logging
import os
import pandas as pd
import numpy as np
import argparse
import json
import ast
import yaml
from joblib import dump
from resource import getrusage, RUSAGE_SELF
from scipy import stats


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)
dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path
agrs_list = ['including_groups', 'grouping_type', 'flow_handling',
            'number_of_intervals', 'encoding', 'output_column',
            'outliers_removal', 'balanced_dataset', 'how_balance',
            'dimensionality_reduction', 'dimensionality_reduction_method',
            'balanced_splitting', 'before_2005', 'data_driven_model']

def isnot_string(val):
    '''
    Function to verify if it is a string
    '''

    try:
        int(float(val))
        return True
    except:
        return False


def to_numeric(val):
    '''
    Function to convert string to numeric
    '''

    val = str(val)
    try:
        return int(val)
    except ValueError:
        return float(val)


def checking_boolean(val):
    '''
    Function to check boolean values
    '''

    if val == 'True':
        return True
    elif val == 'False':
        return False
    else:
        return val

def calc_distance(dimensionality_reduction_method,
                dimensionality_reduction,
                X, feature_cols, num_cols,
                c_centroid=True, centroid=None):

    if (not dimensionality_reduction) or (dimensionality_reduction_method != 'PCA'):
        if c_centroid:
            # Ahmad & dey’s distance
            print(X.shape)
            col_central = list(np.mean(X[:, [i for i, col in enumerate(feature_cols) if col in num_cols]], axis=0))
            col_central_type = ['mean'] * len(col_central)
            index = [i for i, col in enumerate(feature_cols) if col in num_cols]
            col_central = col_central + list(stats.mode(X[:, [i for i, col in enumerate(feature_cols) if not col in num_cols]], axis=0).mode)
            col_central_type = col_central_type + ['mode']*len(set(feature_cols) - set(num_cols))
            index = index +  [i for i, col in enumerate(feature_cols) if col not in num_cols]
            print(len(col_central), len(col_central_type))
            centroid = pd.DataFrame({'centroid': col_central, 'central_tendency': col_central_type})
            print(centroid)
            numerical_vals = centroid.loc[centroid['central_tendency'] == 'mean', 'centroid'].index.tolist()
            distance_numerical = np.sum((X[:, numerical_vals] - centroid.iloc[numerical_vals, 'centroid']) ** 2, axis=1) ** 0.5
            categorical_vals = centroid.loc[centroid['central_tendency'] == 'mode', 'centroid'].index.tolist()
            func = lambda x: 1 if not x else 0
            func = np.vectorize(func)
            matrix_caterorical = X[:, categorical_vals] == centroid.iloc[categorical_vals, 'centroid']
            matrix_caterorical = func(matrix_caterorical)
            distance_categorical = np.sum(matrix_caterorical, axis=1)
            print(distance_numerical.shape, distance_categorical.shape)
            distances = distance_numerical + distance_categorical
            return distances, centroid
        else:
            numerical_vals = centroid.loc[centroid['central_tendency'] == 'mean', 'centroid'].index.tolist()
            distance_numerical = np.sum((X[:, numerical_vals] - centroid.iloc[numerical_vals, 'centroid']) ** 2, axis=1) ** 0.5
            categorical_vals = centroid.loc[centroid['central_tendency'] == 'mode', 'centroid'].index.tolist()
            func = lambda x: 1 if not x else 0
            func = np.vectorize(func)
            matrix_caterorical = X[:, categorical_vals] == centroid.iloc[categorical_vals, 'centroid']
            matrix_caterorical = func(matrix_caterorical)
            distance_categorical = np.sum(matrix_caterorical, axis=1)
            distances = distance_numerical + distance_categorical
            return distances
    else:
        if c_centroid:
            # Euclidean distance
            centroid = pd.DataFrame({'centroid': np.mean(X, axis=0), 'central_tendency': ['mean']*X.shape[1]})
            print(X.shape, centroid.shape)
            distances = np.sum((X - centroid['centroid'].T) ** 2, axis=1) ** 0.5
            return distances, centroid
        else:
            return np.sum((X - centroid['centroid'].T) ** 2, axis=1) ** 0.5
            

def machine_learning_pipeline(args):
    '''
    Function for creating the Machine Learning pipeline
    '''

    if args.input_file == 'No':

        logger = logging.getLogger(' Data-driven modeling')
         
        logger.info(f' Starting data-driven modeling for steps id {args.id}')

        args.model_params = {par: to_numeric(val) if isnot_string(str(val)) else (None if str(val) == 'None' else val) for par, val in json.loads(args.model_params).items()}
        args.model_params = {par: checking_boolean(val) for par, val in args.model_params.items()}
        args.model_params_for_tuning = json.loads(args.model_params_for_tuning)
        args.model_params_for_tuning = {par: ast.literal_eval(val) for par, val in args.model_params_for_tuning.items()}
        args_dict = vars(args)
        args_dict.update({par: checking_boolean(val) for par, val in args_dict.items()})

    else:

        # Selecting data preprocessing params

        ## Opening file for data preprocesing params
        input_file_path = f'{dir_path}/modeling/output/evaluation_output.xlsx'
        input_parms = pd.read_excel(input_file_path,
                                sheet_name='Sheet1',
                                header=None,
                                skiprows=[0, 1, 2],
                                usecols=range(30))

        ## Opening file for data-driven model params
        params_file_path = f'{dir_path}/modeling/input/model_params.yaml'
        with open(params_file_path, mode='r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        myworkbook = openpyxl.load_workbook(input_file_path)
        worksheet = myworkbook['Sheet1']
    
        step_old = 1

        for i, vals in input_parms.iterrows():

            step_new = vals[2]
            run = vals[1]

            if step_new != step_old:
                pos_steps = list(range(1, step_old + 1))
                # FAHP to rank the previous combinations
                rank = data_driven_models_ranking(
                    input_parms.loc[input_parms[2].isin(pos_steps), 
                                    [17, 18, 20, 24, 25, 26, 27]]
                                        ).tolist()
                best = rank.index(min(rank))

                # Looking for column numbers
                cols = [j for j, val in enumerate(input_parms.iloc[i, 0:17].isnull()) if val]
                # Allocating the value
                for col in cols:
                    input_val = input_parms.iloc[best, col]
                    input_parms.iloc[i:, col] = input_val
                    if input_val == True:
                        input_val = 'True'
                    elif input_val == False:
                        input_val = 'False'
                    else:
                        pass
                    for row in list(range(i, input_parms.shape[0])):
                        worksheet.cell(row=row+4, column=col+1).value = input_val
                
                step_old = step_new


            vals = input_parms.iloc[i]
            args_dict = vars(args)
            args_dict.update({par: int(vals[idx+3]) if isnot_string(str(vals[idx+3])) else (None if str(vals[idx+3]) == 'None' else vals[idx+3]) for idx, par in enumerate(agrs_list)})
            if params['model'][args.data_driven_model]['model_params']['defined']:
                model_params = params['model'][args.data_driven_model]['model_params']['defined']
            else:
                model_params = params['model'][args.data_driven_model]['model_params']['default']
            model_params = {par: to_numeric(val) if isnot_string(str(val)) else (None if str(val) == 'None' else val) for par, val in model_params.items()}
            model_params = {par: checking_boolean(val) for par, val in model_params.items()}
            args_dict.update({'id': vals[0],
                            'model_params': model_params})

            if run == 'No':

                logger = logging.getLogger(' Data-driven modeling')

                logger.info(f' Starting data-driven modeling for steps id {args.id}')

                start_time = time.time()

                ## Calling the data preparation pipeline
                data = data_preparation_pipeline(args)

                ## Data
                X_train = data['X_train']
                Y_train = data['Y_train']
                X_test = data['X_test']
                Y_test = data['Y_test']
                del data

                ## Modeling pipeline
                if args.data_driven_model == 'ANNC':
                    args.model_params.update({'input_shape': X_train.shape[1],
                                              'output_shape': len(np.unique(Y_train))})
                modeling_results = modeling_pipeline(X_train, Y_train, 
                                            args.data_driven_model,
                                            args.model_params,
                                            return_model=False)


                running_time = round(time.time() - start_time, 2)
                data_volume = round((X_train.nbytes + X_test.nbytes + Y_train.nbytes + Y_test.nbytes)* 10 ** -9, 2)
                sample_size = X_train.shape[0] + X_test.shape[0]

                input_parms.iloc[i, 17] = modeling_results['balanced_accuracy_validation']
                input_parms.iloc[i, 18] = modeling_results['balanced_accuracy_train']
                input_parms.iloc[i, 19] = modeling_results['balanced_accuracy_analysis']
                input_parms.iloc[i, 20] = modeling_results['error_validation']
                input_parms.iloc[i, 21] = modeling_results['std_error_validation']
                input_parms.iloc[i, 22] = modeling_results['error_train']
                input_parms.iloc[i, 23] = modeling_results['std_error_train']
                input_parms.iloc[i, 24] = modeling_results['mean_y_randomization_error']
                input_parms.iloc[i, 25] = modeling_results['std_y_randomization_error']
                input_parms.iloc[i, 26] = running_time
                input_parms.iloc[i, 27] = data_volume
                input_parms.iloc[i, 28] = sample_size

                ## Saving
                worksheet[f'B{i + 4}'].value = 'Yes'
                worksheet[f'R{i + 4}'].value = modeling_results['balanced_accuracy_validation']
                worksheet[f'S{i + 4}'].value = modeling_results['balanced_accuracy_train']
                worksheet[f'T{i + 4}'].value = modeling_results['balanced_accuracy_analysis']
                worksheet[f'U{i + 4}'].value = modeling_results['error_validation']
                worksheet[f'V{i + 4}'].value = modeling_results['std_error_validation']
                worksheet[f'W{i + 4}'].value = modeling_results['error_train']
                worksheet[f'X{i + 4}'].value = modeling_results['std_error_train']
                worksheet[f'Y{i + 4}'].value = modeling_results['mean_y_randomization_error']
                worksheet[f'Z{i + 4}'].value = modeling_results['std_y_randomization_error']
                worksheet[f'AA{i + 4}'].value = running_time
                worksheet[f'AB{i + 4}'].value = data_volume
                worksheet[f'AC{i + 4}'].value = sample_size

                myworkbook.save(input_file_path)

            else:

                step_old = vals[2]

        # Selecting model

        logger = logging.getLogger(' Data-driven modeling --> Selection')
        logger.info(f' Selecting the model with the best performance based on FAHP')


        input_parms['rank'] = data_driven_models_ranking(
                        input_parms[[17, 18, 20, 24, 25, 26, 27]]
                                            )
        for idx, row in input_parms.iterrows():
            worksheet[f'AD{idx + 4}'].value = row['rank']
        myworkbook.save(input_file_path)
        input_parms = input_parms[input_parms['rank'] == 1]
        input_parms.drop_duplicates(keep='last', subset=[16],
                                        inplace=True)
        if input_parms.shape[0] != 1:
            complexity = {'DTC': 1, 'RFC': 2, 'GBC': 3, 'ANNC': 4}
            input_parms['complexity'] = input_parms[16].apply(lambda x: complexity[x])
            input_parms = input_parms[input_parms.complexity == input_parms.complexity.min()]
        
        vals = input_parms.values[0]
        args_dict = vars(args)
        args_dict.update({par: int(vals[idx+3]) if isnot_string(str(vals[idx+3])) else (None if str(vals[idx+3]) == 'None' else vals[idx+3]) for idx, par in enumerate(agrs_list)})
        if params['model'][args.data_driven_model]['model_params']['defined']:
            model_params = params['model'][args.data_driven_model]['model_params']['defined']
        else:
            model_params = params['model'][args.data_driven_model]['model_params']['default']
        model_params = {par: to_numeric(val) if isnot_string(str(val)) else (None if str(val) == 'None' else val) for par, val in model_params.items()}
        model_params = {par: checking_boolean(val) for par, val in model_params.items()}
        model_params_for_tuning = params['model'][args.data_driven_model]['model_params']['for_tuning']
        func_flatten = lambda t: [item for sublist in t for item in sublist]
        model_params_for_tuning = {k: v if not isinstance(v, dict) else func_flatten([[int(x) if isinstance(vv['start'], int) else round(x, 5) for x in np.linspace(**vv)] if 'default' not in kk else [vv] for kk, vv in v.items()]) for k, v in model_params_for_tuning.items()}
        args_dict.update({'id': vals[0],
                        'model_params': model_params,
                        'save_info': 'Yes',
                        'model_params_for_tuning': model_params_for_tuning})

        logger.info(f' The model {args.data_driven_model} and data preparation id {args.id} were selected ')


    # Calling the data preparation pipeline
    data = data_preparation_pipeline(args)

    # Data
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    feature_cols = data['feature_cols']
    num_cols = data['num_cols']
    del data

    # Training data centroid
    distances_train, centroid = calc_distance(args.dimensionality_reduction_method,
                                            args.dimensionality_reduction,
                                            X_train, feature_cols, num_cols,
                                            c_centroid=True)
    q25, q50, q75 = np.quantile(distances_train, [0.25, 0.5, 0.75])
    iqr = q75 - q25
    cut_off_threshold = q50 + 1.5 * iqr
    if args.save_info == 'Yes':
        centroid.to_csv(f'{dir_path}/modeling/output/data_centroid_id_{args.id}.csv')

    # Including params for ANNC
    if args.data_driven_model == 'ANNC':
        args.model_params.update({'input_shape': X_train.shape[1],
                                  'output_shape': len(np.unique(Y_train))})
    
    if "False" in args.model_params_for_tuning.keys():

        # Fitting the selected model with params
        modeling_results, classifier = modeling_pipeline(X_train, Y_train,
                                    args.data_driven_model,
                                    args.model_params,
                                    return_model=True)
        for key, val in modeling_results.items():
            logger.info(f' The {args.data_driven_model} model {key.replace("_", " ")}: {val}')

    else:
        # Tuning parameters for select model
        fixed_params = {k: v for k, v in args.model_params.items() if k not in  args.model_params_for_tuning}
        logger = logging.getLogger(' Data-driven modeling --> Tuning')
        logger.info(f' Applying randomized grid search for {args.data_driven_model} model')
        fixed_params = {p: v for p, v in args.model_params.items() if p not in args.model_params_for_tuning.keys()}
        results, running_time, classifier = parameter_tuning(X_train, Y_train,
                                                    args.data_driven_model,
                                                    fixed_params,
                                                    args.model_params_for_tuning)
        results = pd.DataFrame(results)
        cols_report = ['mean_fit_time', 'std_fit_time',
                        'mean_test_balanced_accuracy',
                        'std_test_balanced_accuracy',
                        'mean_train_balanced_accuracy',
                        'std_train_balanced_accuracy',
                        'mean_test_accuracy',
                        'std_test_accuracy',
                        'mean_train_accuracy',
                        'std_train_accuracy',
                        'rank_test_accuracy'] + [col for col in results.columns if col.startswith('param_')]
        results = results[cols_report]
        results.sort_values(by=['rank_test_accuracy'],
                            inplace=True)
        results.to_excel(f'{dir_path}/modeling/output/parameters_tuning_id_{args.id}.xlsx',
                        index=False)

    # Evaluating the selected model
    error = prediction_evaluation(classifier, X_test, Y_test, metric='error')
    logger.info(f' Testing the {args.data_driven_model} model on the test set. The {args.data_driven_model} model error: {error}')
    distances_test = np.sum((X_test - centroid) ** 2, axis=1) ** 0.5
    n_outside = X_test[distances_test > cut_off_threshold].shape[0]
    logger.info(f' Number of test samples oustide the applicability domain: {n_outside}')
    error_outside = prediction_evaluation(classifier,
                                X_test[distances_test > cut_off_threshold],
                                Y_test[distances_test > cut_off_threshold],
                                metric='error')
    logger.info(f' Testing the {args.data_driven_model} model on the test samples oustide the applicability domain. The {args.data_driven_model} model error: {error_outside}')
    n_inside = X_test.shape[0] - n_outside
    logger.info(f' Number of test samples inside the applicability domain: {n_inside}')
    error_inside = prediction_evaluation(classifier,
                                X_test[distances_test <= cut_off_threshold],
                                Y_test[distances_test <= cut_off_threshold],
                                metric='error')
    logger.info(f' Testing the {args.data_driven_model} model on the test samples inside the applicability domain. The {args.data_driven_model} model error: {error_inside}')
    if args.save_info == 'Yes':
        result = pd.Series({'Model error on test set': error,
                            'Number of test samples outside the applicability domain': n_outside,
                            'Model error on test samples outside the applicability domain': error_outside,
                            'Number of test samples inside the applicability domain': n_inside,
                            'Model error on test samples inside the applicability domain': error_inside,
                            'Distance threshold for applicability domain': round(cut_off_threshold, 4)})
        result.to_csv(f'{dir_path}/modeling/output/test_error_analysis_{args.id}.xlsx',
                    header=False)

    # Persisting the selected model
    if args.save_info == 'Yes':
       dump(classifier, f'{dir_path}/modeling/output/estimator_id_{args.id}.joblib') 




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rdbms',
                        help='The Relational Database Management System (RDBMS) you would like to use',
                        choices=['mysql', 'postgresql'],
                        type=str,
                        default='mysql')
    parser.add_argument('--password',
                        help='The password for using the RDBMS',
                        type=str)
    parser.add_argument('--username',
                        help='The username for using the RDBMS',
                        type=str,
                        default='root')
    parser.add_argument('--host',
                        help='The computer hosting for the database',
                        type=str,
                        default='127.0.0.1')
    parser.add_argument('--port',
                        help='Port used by the database engine',
                        type=str,
                        default='3306')
    parser.add_argument('--db_name',
                        help='Database name',
                        type=str,
                        default='PRTR_transfers')
    parser.add_argument('--including_groups',
                        help='Would you like to include the chemical groups',
                        choices=['True', 'False'],
                        type=str,
                        default='True')
    parser.add_argument('--grouping_type',
                        help='How you want to calculate descriptors for the chemical groups',
                        choices=[1, 2, 3, 4, 5, 6, 7, 8],
                        type=int,
                        required=False,
                        default=1)
    parser.add_argument('--flow_handling',
                        help='How you want to handle the transfer flow rates',
                        choices=[1, 2, 3, 4],
                        type=int,
                        required=False,
                        default=1)
    parser.add_argument('--number_of_intervals',
                        help='How many intervals would you like to use for the transfer flow rates',
                        type=int,
                        required=False,
                        default=10)
    parser.add_argument('--encoding',
                        help='How you want to encode the non-ordinal categorical data',
                        choices=['one-hot-encoding', 'target-encoding'],
                        type=str,
                        required=False,
                        default='one-hot-encoding')
    parser.add_argument('--output_column',
                        help='What column would you like to keep as the classifier output',
                        choices=['generic', 'wm_hierarchy'],
                        type=str,
                        required=False,
                        default='generic')
    parser.add_argument('--outliers_removal',
                        help='Would you like to keep the outliers',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--balanced_dataset',
                        help='Would you like to balance the dataset',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--how_balance',
                        help='What method to balance the dataset you would like to use (see options)',
                        choices=['random_oversample', 'smote', 'adasyn', 'random_undersample', 'near_miss'],
                        type=str,
                        required=False,
                        default='random_oversample')
    parser.add_argument('--dimensionality_reduction',
                        help='Would you like to apply dimensionality reduction?',
                        choices=['False',  'True'],
                        type=str,
                        required=False,
                        default='False')
    parser.add_argument('--dimensionality_reduction_method',
                        help='What method for dimensionality reduction would you like to apply?. In this point, after encoding, we only apply feature transformation by PCA - Principal Component Analysis or feature selection by UFS - Univariate Feature Selection with mutual information metric or RFC - Random Forest Classifier',
                        choices=['PCA', 'UFS', 'RFC'],
                        type=str,
                        required=False,
                        default='PCA')
    parser.add_argument('--balanced_splitting',
                        help='Would you like to split the dataset in a balanced fashion',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--before_2005',
                        help='Would you like to include data reported before 2005?',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--input_file',
                        help='Do you have an input file?',
                        choices=['Yes', 'No'],
                        type=str,
                        required=False,
                        default='No')
    parser.add_argument('--save_info',
                        help='Would you like to save information?',
                        choices=['Yes', 'No'],
                        type=str,
                        required=False,
                        default='No')
    parser.add_argument('--data_driven_model',
                        help='What classification model would you like to use?',
                        choices=['DTC', 'RFC', 'GBC', 'ANNC'],
                        type=str,
                        required=False,
                        default='DTC')
    parser.add_argument('--id',
                        help='What id whould your like to use',
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument('--model_params',
                        help='What params would you like to use for fitting the model',
                        type=str,
                        required=False,
                        default='{"random_state": 0}')
    parser.add_argument('--model_params_for_tuning',
                        help='What params would you like to use for tuning the model',
                        type=str,
                        required=False,
                        default='{"False": "[1, None]"}')
    

    args = parser.parse_args()

    start_simulation = time.time()
    machine_learning_pipeline(args)
    simulation_running_time = round(time.time() - start_simulation, 2)

    print(f'Simulation time [seg]: {simulation_running_time}')
    print("Peak memory (MiB):", int(getrusage(RUSAGE_SELF).ru_maxrss / 1024))