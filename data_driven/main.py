#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

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
            'number_of_intervals', 'output_column',
            'outliers_removal', 'balanced_dataset',
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
                                usecols=range(29))
        input_parms[[0, 2]] = input_parms[[0, 2]].astype(int) 
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
                                    [16, 18, 19, 23, 24, 25, 26]]
                                        ).tolist()
                best = rank.index(min(rank))

                # Looking for column numbers
                cols = [j for j, val in enumerate(input_parms.iloc[i, 0:16].isnull()) if val]
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
                if args.data_driven_model == 'ANNC':
                    n_inputs, n_outputs = X_train.shape[1], Y_train.shape[1]
                    args.model_params.update({'input_shape': n_inputs,
                                              'output_shape': n_outputs,
                                              'units_per_layer': (int(np.mean([n_inputs, n_outputs])),)})
                modeling_results = modeling_pipeline(X_train, Y_train, 
                                            args.data_driven_model,
                                            args.model_params,
                                            return_model=False)


                running_time = round(time.time() - start_time, 2)
                data_volume = round((X_train.nbytes + X_test.nbytes + Y_train.nbytes + Y_test.nbytes)* 10 ** -9, 2)
                sample_size = X_train.shape[0] + X_test.shape[0]

                input_parms.iloc[i, 16] = modeling_results['mean_validation_acc']
                input_parms.iloc[i, 17] = modeling_results['mean_train_acc']
                input_parms.iloc[i, 18] = modeling_results['accuracy_analysis']
                input_parms.iloc[i, 19] = modeling_results['mean_validation_hamming_loss']
                input_parms.iloc[i, 20] = modeling_results['std_validation_hamming_loss']
                input_parms.iloc[i, 21] = modeling_results['mean_train_hamming_loss']
                input_parms.iloc[i, 22] = modeling_results['std_train_hamming_loss']
                input_parms.iloc[i, 23] = modeling_results['mean_y_randomization_hamming_loss']
                input_parms.iloc[i, 24] = modeling_results['std_y_randomization_hamming_loss']
                input_parms.iloc[i, 25] = running_time
                input_parms.iloc[i, 26] = data_volume
                input_parms.iloc[i, 27] = sample_size

                ## Saving
                worksheet[f'B{i + 4}'].value = 'Yes'
                worksheet[f'Q{i + 4}'].value = modeling_results['mean_validation_acc']
                worksheet[f'R{i + 4}'].value = modeling_results['mean_train_acc']
                worksheet[f'S{i + 4}'].value = modeling_results['accuracy_analysis']
                worksheet[f'T{i + 4}'].value = modeling_results['mean_validation_hamming_loss']
                worksheet[f'U{i + 4}'].value = modeling_results['std_validation_hamming_loss']
                worksheet[f'V{i + 4}'].value = modeling_results['mean_train_hamming_loss']
                worksheet[f'W{i + 4}'].value = modeling_results['std_train_hamming_loss']
                worksheet[f'X{i + 4}'].value = modeling_results['mean_y_randomization_hamming_loss']
                worksheet[f'Y{i + 4}'].value = modeling_results['std_y_randomization_hamming_loss']
                worksheet[f'Z{i + 4}'].value = running_time
                worksheet[f'AA{i + 4}'].value = data_volume
                worksheet[f'AB{i + 4}'].value = sample_size

                myworkbook.save(input_file_path)

            else:

                step_old = vals[2]

        # Selecting model

        logger = logging.getLogger(' Data-driven modeling --> Selection')
        logger.info(f' Selecting the model with the best performance based on FAHP')


        input_parms['rank'] = data_driven_models_ranking(
                        input_parms[[16, 18, 19, 23, 24, 25, 26]]
                                            )
        for idx, row in input_parms.iterrows():
            worksheet[f'AC{idx + 4}'].value = row['rank']
        myworkbook.save(input_file_path)
        input_parms = input_parms[input_parms['rank'] == 1]
        input_parms.drop_duplicates(keep='last', subset=[15],
                                        inplace=True)
        if input_parms.shape[0] != 1:
            complexity = {'DTC': 1, 'RFC': 2, 'GBC': 3, 'ANNC': 4}
            input_parms['complexity'] = input_parms[15].apply(lambda x: complexity[x])
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
    if args.data_driven_model == 'ANNC':
        n_inputs, n_outputs = X_train.shape[1], Y_train.shape[1]
        args.model_params.update({'input_shape': n_inputs,
                                  'output_shape': n_outputs})
    
    if "False" in args.model_params_for_tuning.keys():

        # Fitting the selected model with params
        modeling_results, classifier = modeling_pipeline(X_train, Y_train,
                                    args.data_driven_model,
                                    args.model_params,
                                    return_model=True)
        logger = logging.getLogger(' Data-driven modeling --> Evaluation')
        for key, val in modeling_results.items():
            logger.info(f' The {args.data_driven_model} model {key.replace("_", " ")}: {val}')

    else:
        # Tuning parameters for select model
        logger = logging.getLogger(' Data-driven modeling --> Tuning')
        logger.info(f' Applying randomized grid search for {args.data_driven_model} model')
        classifier, df_tuning, df_model_params = parameter_tuning(X_train, Y_train,
                                                    args.data_driven_model,
                                                    args.model_params,
                                                    args.model_params_for_tuning)

        if args.save_info:

            df_tuning.to_csv(f'{dir_path}/modeling/output/tuning_result_id_{args.id}.csv', index=False)
            df_model_params.to_csv(f'{dir_path}/modeling/output/tuning_best_params_id_{args.id}.csv')

        
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
    loss_test = prediction_evaluation(classifier=classifier, X=X_test, Y=Y_test, metric='hamming_loss_score')
    acc_test = prediction_evaluation(classifier=classifier, X=X_test, Y=Y_test)
    logger.info(f' Testing the {args.data_driven_model} model on the test set. The {args.data_driven_model} model hamming loss: {loss_test}')
    logger.info(f' Testing the {args.data_driven_model} model on the test set. The {args.data_driven_model} model accuracy: {acc_test}')
    n_outside = X_test[distances_test > cut_off_threshold].shape[0]
    logger.info(f' Number of test samples oustide the applicability domain: {n_outside}')
    loss_test_outside = prediction_evaluation(classifier=classifier,
                                X=X_test[distances_test > cut_off_threshold],
                                Y=Y_test[distances_test > cut_off_threshold],
                                metric='hamming_loss_score')
    acc_test_outside = prediction_evaluation(classifier=classifier,
                                X=X_test[distances_test > cut_off_threshold],
                                Y=Y_test[distances_test > cut_off_threshold])
    logger.info(f' Testing the {args.data_driven_model} model on the test samples oustide the applicability domain. The {args.data_driven_model} model hamming loss: {loss_test_outside}')
    logger.info(f' Testing the {args.data_driven_model} model on the test samples oustide the applicability domain. The {args.data_driven_model} model accuracy: {acc_test_outside}')
    n_inside = X_test.shape[0] - n_outside
    logger.info(f' Number of test samples inside the applicability domain: {n_inside}')
    loss_test_inside = prediction_evaluation(classifier=classifier,
                                X=X_test[distances_test <= cut_off_threshold],
                                Y=Y_test[distances_test <= cut_off_threshold],
                                metric='hamming_loss_score')
    acc_test_inside = prediction_evaluation(classifier=classifier,
                                X=X_test[distances_test <= cut_off_threshold],
                                Y=Y_test[distances_test <= cut_off_threshold])
    logger.info(f' Testing the {args.data_driven_model} model on the test samples inside the applicability domain. The {args.data_driven_model} model hamming loss: {loss_test_inside}')
    logger.info(f' Testing the {args.data_driven_model} model on the test samples inside the applicability domain. The {args.data_driven_model} model accuracy: {acc_test_inside}')
    n_test_equal_to_train = (X_train == X_test[:, None]).all(axis=2).any(axis=1).sum()
    logger.info(f' The number of test examples whose predictors (Xs) are in the training data is {n_test_equal_to_train} ')
    if args.save_info == 'Yes':
        result = pd.Series({'Model hamming loss on test set': loss_test,
                            'Model accuracy on test set': acc_test,
                            'Number of test samples outside the applicability domain': n_outside,
                            'Model hamming loss on test samples outside the applicability domain': loss_test_outside,
                            'Model accuracy on test samples outside the applicability domain': acc_test_outside,
                            'Number of test samples inside the applicability domain': n_inside,
                            'Model hamming loss on test samples inside the applicability domain': loss_test_inside,
                            'Model accuracy on test samples inside the applicability domain': acc_test_inside,
                            'Distance threshold for applicability domain': round(cut_off_threshold, 4),
                            'Number of test examples whose predictors (Xs) are in the training data': n_test_equal_to_train})
        result.to_csv(f'{dir_path}/modeling/output/test_error_analysis_{args.id}.xlsx',
                    header=False)

    # Persisting the selected model
    if args.save_info == 'Yes':
       pickle.dump(classifier, open(f'{dir_path}/modeling/output/estimator_id_{args.id}.pkl', 'wb'))




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
                        choices=['DTC', 'RFC', 'GBC', 'ANNC'],
                        type=str,
                        required=False,
                        default='DTC')
    parser.add_argument('--id',
                        help='What id whould your like to use for the data preparation workflow',
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
    parser.add_argument('--classification_type',
                        help='What kind of classification problem would you like ',
                        choices=['multi-model binary classification', 'multi-label classification', 'multi-class classification'],
                        type=str,
                        required=True,
                        default='multi-class classification')
    parser.add_argument('--balanaced_split',
                        help='Would you like to obtain an stratified train-test split?',
                        choices=['True', 'False'],
                        type=str,
                        required=True,
                        default='True')
    

    args = parser.parse_args()

    start_simulation = time.time()
    machine_learning_pipeline(args)
    simulation_running_time = round(time.time() - start_simulation, 2)

    print(f'Simulation time [seg]: {simulation_running_time}')
    print("Peak memory (MiB):", int(getrusage(RUSAGE_SELF).ru_maxrss / 1024))