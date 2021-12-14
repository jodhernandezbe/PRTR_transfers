#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

from data_driven.data_preparation.main import data_preparation_pipeline
from data_driven.modeling.main import modeling_pipeline
from data_driven.modeling.evaluation import data_driven_models_ranking, prediction_evaluation, centroid_cal, calc_distance
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
import pickle
from resource import getrusage, RUSAGE_SELF

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)
dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path
agrs_list = ['including_groups', 'grouping_type', 'flow_handling',
            'number_of_intervals', 'encoding', 'output_column',
            'outliers_removal', 'dimensionality_reduction', 'dimensionality_reduction_method',
            'before_2005', 'data_driven_model']

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
                                usecols=range(25))

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
                                    [14, 15, 19, 20, 21, 22]]
                                        ).tolist()
                best = rank.index(min(rank))

                # Looking for column numbers
                cols = [j for j, val in enumerate(input_parms.iloc[i, 0:14].isnull()) if val]
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
            args_dict.update({par: int(vals[idx+3]) if isnot_string(str(vals[idx+3])) else (None if str(vals[idx+3]) == 'None' else vals[idx+3]) for 
            idx, par in enumerate(agrs_list)})
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
                if args.data_driven_model == 'ANNR':
                    n_inputs, n_outputs = X_train.shape[1], Y_train.shape[1]
                    args.model_params.update({'input_shape': n_inputs,
                                              'output_shape': n_outputs})
                modeling_results = modeling_pipeline(X_train, Y_train, 
                                            args.data_driven_model,
                                            args.model_params,
                                            return_model=False)


                running_time = round(time.time() - start_time, 2)
                data_volume = round((X_train.nbytes + X_test.nbytes + Y_train.nbytes + Y_test.nbytes)* 10 ** -9, 2)
                sample_size = X_train.shape[0] + X_test.shape[0]

                input_parms.iloc[i, 14] = modeling_results['mape_analysis']
                input_parms.iloc[i, 15] = modeling_results['mape_validation']
                input_parms.iloc[i, 16] = modeling_results['std_mape_validation']
                input_parms.iloc[i, 17] = modeling_results['mape_train']
                input_parms.iloc[i, 18] = modeling_results['std_mape_train']
                input_parms.iloc[i, 19] = modeling_results['mean_y_randomization_mape']
                input_parms.iloc[i, 20] = modeling_results['std_y_randomization_mape']
                input_parms.iloc[i, 21] = running_time
                input_parms.iloc[i, 22] = data_volume
                input_parms.iloc[i, 23] = sample_size

                ## Saving
                worksheet[f'B{i + 4}'].value = 'Yes'
                worksheet[f'O{i + 4}'].value = modeling_results['mape_analysis']
                worksheet[f'P{i + 4}'].value = modeling_results['mape_validation']
                worksheet[f'Q{i + 4}'].value = modeling_results['std_mape_validation']
                worksheet[f'R{i + 4}'].value = modeling_results['mape_train']
                worksheet[f'S{i + 4}'].value = modeling_results['std_mape_train']
                worksheet[f'T{i + 4}'].value = modeling_results['mean_y_randomization_mape']
                worksheet[f'U{i + 4}'].value = modeling_results['std_y_randomization_mape']
                worksheet[f'V{i + 4}'].value = running_time
                worksheet[f'W{i + 4}'].value = data_volume
                worksheet[f'X{i + 4}'].value = sample_size

                myworkbook.save(input_file_path)

            else:

                step_old = vals[2]

        # Selecting model

        logger = logging.getLogger(' Data-driven modeling --> Selection')
        logger.info(f' Selecting the model with the best performance based on FAHP')


        input_parms['rank'] = data_driven_models_ranking(
                        input_parms[[14, 15, 19, 20, 21, 22]]
                                            )
        for idx, row in input_parms.iterrows():
            worksheet[f'Y{idx + 4}'].value = row['rank']
        myworkbook.save(input_file_path)
        input_parms = input_parms[input_parms['rank'] == 1]
        input_parms.drop_duplicates(keep='last', subset=[13],
                                        inplace=True)
        if input_parms.shape[0] != 1:
            complexity = {'DTR': 1, 'RFR': 2, 'GBR': 3, 'ANNR': 4}
            input_parms['complexity'] = input_parms[13].apply(lambda x: complexity[x])
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

    # Including params for ANNR
    if args.data_driven_model == 'ANNR':
        n_inputs, n_outputs = X_train.shape[1], Y_train.shape[1]
        args.model_params.update({'input_shape': n_inputs,
                                  'output_shape': n_outputs})
    
    if "False" in args.model_params_for_tuning.keys():

        # Fitting the selected model with params
        modeling_results, regressor = modeling_pipeline(X_train, Y_train,
                                    args.data_driven_model,
                                    args.model_params,
                                    return_model=True)
        logger = logging.getLogger(' Data-driven modeling --> Evaluation')
        for key, val in modeling_results.items():
            logger.info(f' The {args.data_driven_model} model {key.replace("_", " ")}: {val}')

    else:
        # Tuning parameters for select model
        fixed_params = {k: v for k, v in args.model_params.items() if k not in  args.model_params_for_tuning}
        logger = logging.getLogger(' Data-driven modeling --> Tuning')
        logger.info(f' Applying randomized grid search for {args.data_driven_model} model')
        fixed_params = {p: v for p, v in args.model_params.items() if p not in args.model_params_for_tuning.keys()}
        results, running_time, regressor = parameter_tuning(X_train, Y_train,
                                                    args.data_driven_model,
                                                    fixed_params,
                                                    args.model_params_for_tuning)
        results = pd.DataFrame(results)
        cols_report = ['mean_fit_time', 'std_fit_time',
                        'mean_test_mean_absolute_percentage_error',
                        'std_test_mean_absolute_percentage_error',
                        'mean_train_mean_absolute_percentage_error',
                        'std_train_mean_absolute_percentage_error',
                        'rank_test_mean_absolute_percentage_error'] + [col for col in results.columns if col.startswith('param_')]
        results = results[cols_report]
        results.sort_values(by=['rank_test_mean_absolute_percentage_error'],
                            inplace=True)
        results.to_excel(f'{dir_path}/modeling/output/parameters_tuning_id_{args.id}.xlsx',
                        index=False)

    # Training data centroid
    logger = logging.getLogger(' Data-driven modeling --> Evaluation')
    logger.info(f' Calculating the training data centroid')
    centroid = centroid_cal(args.dimensionality_reduction_method,
                            args.dimensionality_reduction,
                            X_train, feature_cols, num_cols)
    if args.save_info == 'Yes':
        centroid.to_csv(f'{dir_path}/modeling/output/data_centroid_id_{args.id}.csv')

    # Threshold distance
    logger.info(f' Calculating the threshold distance')
    distances_train = calc_distance(args.dimensionality_reduction_method,
                                    args.dimensionality_reduction,
                                    X_train, centroid)
    q1, q2, q3 = np.quantile(distances_train, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    cut_off_threshold = q2 + 1.5*iqr

    # Calculating distances for the test data to the traing data centroid
    logger.info(f' Calculating distances for the test data to the traing data centroid')
    distances_test = calc_distance(args.dimensionality_reduction_method,
                                    args.dimensionality_reduction,
                                    X_test, centroid)

    # Evaluating the selected model
    mape_test = prediction_evaluation(regressor, X_test, Y_test)
    logger.info(f' Testing the {args.data_driven_model} model on the test set. The {args.data_driven_model} model MAPE: {mape_test}')
    n_outside = X_test[distances_test > cut_off_threshold].shape[0]
    logger.info(f' Number of test samples oustide the applicability domain: {n_outside}')
    mape_test_outside = prediction_evaluation(regressor,
                                X_test[distances_test > cut_off_threshold],
                                Y_test[distances_test > cut_off_threshold])
    logger.info(f' Testing the {args.data_driven_model} model on the test samples oustide the applicability domain. The {args.data_driven_model} model MAPE: {mape_test_outside}')
    n_inside = X_test.shape[0] - n_outside
    logger.info(f' Number of test samples inside the applicability domain: {n_inside}')
    mape_test_inside = prediction_evaluation(regressor,
                                X_test[distances_test <= cut_off_threshold],
                                Y_test[distances_test <= cut_off_threshold])
    logger.info(f' Testing the {args.data_driven_model} model on the test samples inside the applicability domain. The {args.data_driven_model} model MAPE: {mape_test_inside}')
    n_test_equal_to_train = (X_train == X_test[:, None]).all(axis=2).any(axis=1).sum()
    logger.info(f' The number of test examples whose predictors (Xs) are in the training data is {n_test_equal_to_train} ')
    if args.save_info == 'Yes':
        result = pd.Series({'Model MAPE on test set': mape_test,
                            'Number of test samples outside the applicability domain': n_outside,
                            'Model MAPE on test samples outside the applicability domain': mape_test_outside,
                            'Number of test samples inside the applicability domain': n_inside,
                            'Model error on test samples inside the applicability domain': mape_test_inside,
                            'Distance threshold for applicability domain': round(cut_off_threshold, 4)})
        result.to_csv(f'{dir_path}/modeling/output/test_error_analysis_{args.id}.xlsx',
                    header=False)

    # Persisting the selected model
    if args.save_info == 'Yes':
       pickle.dump(regressor, open(f'{dir_path}/modeling/output/estimator_id_{args.id}.pkl', 'wb')) 




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
                        help='What column would you like to keep as the regressor output',
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
    parser.add_argument('--dimensionality_reduction',
                        help='Would you like to apply dimensionality reduction?',
                        choices=['False',  'True'],
                        type=str,
                        required=False,
                        default='False')
    parser.add_argument('--dimensionality_reduction_method',
                        help='What method for dimensionality reduction would you like to apply?. In this point, after encoding, we only apply feature transformation by PCA - Principal Component Analysis or feature selection by UFS - Univariate Feature Selection with mutual infomation (filter method) or RFR - Random Forest Regressor (embedded method via feature importance)',
                        choices=['PCA', 'UFS', 'RFR'],
                        type=str,
                        required=False,
                        default='PCA')
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
                        help='What regression model would you like to use?',
                        choices=['DTR', 'RFR', 'GBR', 'ANNR'],
                        type=str,
                        required=False,
                        default='DTR')
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