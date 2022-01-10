#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

from data_driven.data_preparation.main import data_preparation_pipeline
from data_driven.modeling.main import modeling_pipeline
from data_driven.modeling.evaluation import prediction_evaluation, centroid_cal, calc_distance, external_evaluation
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
agrs_list = ['before_2005', 'including_groups', 'grouping_type',
            'flow_handling', 'number_of_intervals', 'output_column',
            'outliers_removal', 'balanced_splitting', 'dimensionality_reduction',
            'dimensionality_reduction_method', 'balanced_dataset', 'classification_type',
            'target_class', 'data_driven_model']


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
         
        logger.info(f' Starting data-driven modeling for ML trail id {args.trial_id}')

        args.model_params = {par: to_numeric(val) if isnot_string(str(val)) else (None if str(val) == 'None' else val) for par, val in json.loads(args.model_params).items()}
        args.model_params = {par: checking_boolean(val) for par, val in args.model_params.items()}
        args.model_params_for_tuning = json.loads(args.model_params_for_tuning)
        args.model_params_for_tuning = {par: ast.literal_eval(val) for par, val in args.model_params_for_tuning.items()}
        args_dict = vars(args)
        args_dict.update({par: checking_boolean(val) for par, val in args_dict.items()})

    else:

        ## Opening file for data preprocesing params
        input_file_path = f'{dir_path}/modeling/output/evaluation_output.xlsx'
        input_parms = pd.read_excel(input_file_path,
                                sheet_name='classification',
                                header=None,
                                skiprows=[0, 1],
                                usecols=range(41))
        input_parms[[0, 2]] = input_parms[[0, 2]].astype(int)
        myworkbook = openpyxl.load_workbook(input_file_path)
        worksheet = myworkbook['classification']

        ## Opening file for data-driven model params
        params_file_path = f'{dir_path}/modeling/input/model_params.yaml'
        with open(params_file_path, mode='r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    
        for i, vals in input_parms.iterrows():

            ## Organizing input params
            run = vals[1]
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
            args_dict.update({'id': vals[2],
                            'trial_id': vals[0], 
                            'model_params': model_params,
                            'save_info': 'Yes'})

            if run == 'No':

                logger = logging.getLogger(' Data-driven modeling')
                logger.info(f' Starting data-driven modeling for ML trail id {args.trial_id}')

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
                    n_inputs = X_train.shape[1]
                    if args.classification_type == 'multi-model binary classification':
                        n_outputs = 1
                    elif args.classification_type == 'multi-class classification':
                        n_outputs = len(np.unique(Y_train))
                    else:
                        n_outputs = Y_train.shape[1]
                    args.model_params.update({'input_shape': n_inputs,
                                              'output_shape': n_outputs,
                                              'classification_type': args.classification_type})
                modeling_results, classifier = modeling_pipeline(X_train, Y_train, 
                                            args.data_driven_model,
                                            args.model_params,
                                            args.classification_type,
                                            return_model=True)

                ## External validation
                logger.info(f' Evaluation on test set')
                external_testing_results, evaluation_time, cut_off_threshold, n_outside, n_inside = external_evaluation(X_test, Y_test, X_train, classifier,
                                                                args.dimensionality_reduction_method,
                                                                args.dimensionality_reduction,
                                                                args.classification_type, args.id)
                data_volume = round((X_train.nbytes + X_test.nbytes + Y_train.nbytes + Y_test.nbytes)* 10 ** -9, 2)

                input_parms.iloc[i, 17] = modeling_results['accuracy_analysis']
                input_parms.iloc[i, 18] = modeling_results['mean_validation_accuracy']
                input_parms.iloc[i, 19] = modeling_results['mean_train_accuracy']
                input_parms.iloc[i, 20] = modeling_results['mean_validation_f1']
                input_parms.iloc[i, 21] = modeling_results['mean_train_f1']
                input_parms.iloc[i, 22] = modeling_results['mean_validation_0_1_loss_or_error']
                input_parms.iloc[i, 23] = modeling_results['std_validation_0_1_loss_or_error']
                input_parms.iloc[i, 24] = modeling_results['y_randomization_mean_0_1_loss_or_error']
                input_parms.iloc[i, 25] = modeling_results['y_randomization_std_0_1_loss_or_error']
                input_parms.iloc[i, 26] = modeling_results['y_randomization_analysis']
                input_parms.iloc[i, 27] = external_testing_results['global_accuracy']
                input_parms.iloc[i, 28] = external_testing_results['global_f1']
                input_parms.iloc[i, 29] = external_testing_results['global_hamming_loss_or_error']
                input_parms.iloc[i, 30] = external_testing_results['outside_ad_accuracy']
                input_parms.iloc[i, 31] = external_testing_results['outside_ad_f1']
                input_parms.iloc[i, 32] = external_testing_results['outside_ad_hamming_loss_or_error']
                input_parms.iloc[i, 33] = external_testing_results['inside_ad_accuracy']
                input_parms.iloc[i, 34] = external_testing_results['inside_ad_f1']
                input_parms.iloc[i, 35] = external_testing_results['inside_ad_hamming_loss_or_error']
                input_parms.iloc[i, 36] = evaluation_time
                input_parms.iloc[i, 37] = data_volume
                input_parms.iloc[i, 38] = cut_off_threshold
                input_parms.iloc[i, 39] = n_outside
                input_parms.iloc[i, 40] = n_inside

                ## Saving
                worksheet[f'B{i + 3}'].value = 'Yes'
                worksheet[f'R{i + 3}'].value = modeling_results['accuracy_analysis']
                worksheet[f'S{i + 3}'].value = modeling_results['mean_validation_accuracy']
                worksheet[f'T{i + 3}'].value = modeling_results['mean_train_accuracy']
                worksheet[f'U{i + 3}'].value = modeling_results['mean_validation_f1']
                worksheet[f'V{i + 3}'].value = modeling_results['mean_train_f1']
                worksheet[f'W{i + 3}'].value = modeling_results['mean_validation_0_1_loss_or_error']
                worksheet[f'X{i + 3}'].value = modeling_results['std_validation_0_1_loss_or_error']
                worksheet[f'Y{i + 3}'].value = modeling_results['y_randomization_mean_0_1_loss_or_error']
                worksheet[f'Z{i + 3}'].value = modeling_results['y_randomization_std_0_1_loss_or_error']
                worksheet[f'AA{i + 3}'].value = modeling_results['y_randomization_analysis']
                worksheet[f'AB{i + 3}'].value = external_testing_results['global_accuracy']
                worksheet[f'AC{i + 3}'].value = external_testing_results['global_f1']
                worksheet[f'AD{i + 3}'].value = external_testing_results['global_hamming_loss_or_error']
                worksheet[f'AE{i + 3}'].value = external_testing_results['outside_ad_accuracy']
                worksheet[f'AF{i + 3}'].value = external_testing_results['outside_ad_f1']
                worksheet[f'AG{i + 3}'].value = external_testing_results['outside_ad_hamming_loss_or_error']
                worksheet[f'AH{i + 3}'].value = external_testing_results['inside_ad_accuracy']
                worksheet[f'AI{i + 3}'].value = external_testing_results['inside_ad_f1']
                worksheet[f'AJ{i + 3}'].value = external_testing_results['inside_ad_hamming_loss_or_error']
                worksheet[f'AK{i + 3}'].value = evaluation_time
                worksheet[f'AL{i + 3}'].value = data_volume
                worksheet[f'AM{i + 3}'].value = cut_off_threshold
                worksheet[f'AN{i + 3}'].value = n_outside
                worksheet[f'AO{i + 3}'].value = n_inside
                myworkbook.save(input_file_path)

        # Selecting modeling pipeline by classification
        logger = logging.getLogger(' Data-driven modeling --> Selection')
        logger.info(f' Selecting modeling pipeline by classification using FAHP')

        # Selecting classification modeling strategy
        logger = logging.getLogger(' Data-driven modeling --> Selection')
        logger.info(f' Selecting classification modeling strategy using FAHP')



        #model_params_for_tuning = params['model'][args.data_driven_model]['model_params']['for_tuning']
        #func_flatten = lambda t: [item for sublist in t for item in sublist]
        #model_params_for_tuning = {k: v if not isinstance(v, dict) else func_flatten([[int(x) if isinstance(vv['start'], int) else round(x, 5) for x in np.linspace(**vv)] if 'default' not in kk else [vv] for kk, vv in v.items()]) for k, v in model_params_for_tuning.items()}
        #args_dict.update({'id': vals[0],
        #                'model_params': model_params,
        #                'save_info': 'Yes',
        #                'model_params_for_tuning': model_params_for_tuning})
        #
        #logger.info(f' The model {args.data_driven_model} and data preparation id {args.id} were selected ')

    # Calling the data preparation pipeline
    #data = data_preparation_pipeline(args)

    # Data
    #X_train = data['X_train']
    #Y_train = data['Y_train']
    #X_test = data['X_test']
    #Y_test = data['Y_test']
    #del data

    # Including params for ANNR
    #if args.data_driven_model == 'ANNC':
    #    n_inputs, n_outputs = X_train.shape[1], Y_train.shape[1]
    #    args.model_params.update({'input_shape': n_inputs,
    #                              'output_shape': n_outputs})
    
    #if "False" in args.model_params_for_tuning.keys():

        # Fitting the selected model with params
    #    modeling_results, classifier = modeling_pipeline(X_train, Y_train,
    #                                args.data_driven_model,
    #                                args.model_params,
    #                                return_model=True)
    #    logger = logging.getLogger(' Data-driven modeling --> Evaluation')
    #    for key, val in modeling_results.items():
    #        logger.info(f' The {args.data_driven_model} model {key.replace("_", " ")}: {val}')

    #else:
        # Tuning parameters for select model
    #    logger = logging.getLogger(' Data-driven modeling --> Tuning')
    #    logger.info(f' Applying randomized grid search for {args.data_driven_model} model')
    #    classifier, df_tuning, df_model_params = parameter_tuning(X_train, Y_train,
    #                                                args.data_driven_model,
    #                                                args.model_params,
    #                                                args.model_params_for_tuning)

    #    if args.save_info:

    #        df_tuning.to_csv(f'{dir_path}/modeling/output/tuning_result_id_{args.id}.csv', index=False)
    #        df_model_params.to_csv(f'{dir_path}/modeling/output/tuning_best_params_id_{args.id}.csv')

    # Persisting the selected model
    #if args.save_info == 'Yes':
    #   pickle.dump(classifier, open(f'{dir_path}/modeling/output/estimator_id_{args.id}.pkl', 'wb'))




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
    parser.add_argument('--id',
                        help='What id whould your like to use for the data preparation workflow',
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument('--before_2005',
                        help='Would you like to include data reported before 2005?',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
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
    parser.add_argument('--balanaced_split',
                        help='Would you like to obtain an stratified train-test split?',
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
                        help='What method for dimensionality reduction would you like to apply?. In this point, after encoding, we only apply feature transformation by FAMD - Factor Analysis of Mixed Data or feature selection by UFS - Univariate Feature Selection with mutual infomation (filter method) or RFC - Random Forest Classifier (embedded method via feature importance)',
                        choices=['FAMD', 'UFS', 'RFC'],
                        type=str,
                        required=False,
                        default='FAMD')
    parser.add_argument('--balanced_dataset',
                        help='Would you like to balance the dataset',
                        choices=['True', 'False'],
                        type=str,
                        required=False,
                        default='True')
    parser.add_argument('--classification_type',
                        help='What kind of classification problem would you like ',
                        choices=['multi-model binary classification', 'multi-label classification', 'multi-class classification'],
                        type=str,
                        required=False,
                        default='multi-class classification')
    parser.add_argument('--target_class',
                        help='If applied, What is the target class (only for multi-model binary classification)',
                        choices=['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'Disposal', 'Sewerage', 'Treatment', 'Energy recovery', 'Recycling'],
                        type=str,
                        required=False,
                        default='M1')
    parser.add_argument('--data_driven_model',
                        help='What regression model would you like to use?',
                        choices=['RFC', 'ANNC'],
                        type=str,
                        required=False,
                        default='DTC')
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
    parser.add_argument('--trial_id',
                        help='What id whould your like to use for the ML trial',
                        type=int,
                        required=False,
                        default=0)
    

    args = parser.parse_args()

    start_simulation = time.time()
    machine_learning_pipeline(args)
    simulation_running_time = round(time.time() - start_simulation, 2)

    print(f'Simulation time [seg]: {simulation_running_time}')
    print("Peak memory (MiB):", int(getrusage(RUSAGE_SELF).ru_maxrss / 1024))