#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.data_preparation.main import data_preparation_pipeline
from data_driven.modeling.main import modeling_pipeline
from data_driven.modeling.evaluation import data_driven_models_ranking, prediction_evaluation

import time
import openpyxl
import logging
import os
import pandas as pd
import argparse
import json
import yaml

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
            

def machine_learning_pipeline(args):
    '''
    Function for creating the Machine Learning pipeline
    '''

    if args.intput_file == 'No':

        args.model_params = {par: int(val) if isnot_string(str(val)) else (None if str(val) == 'None' else str(val)) for par, val in json.loads(args.model_params).items()}
        args.model_params_for_tuning = {par: int(val) if isnot_string(str(val)) else (None if str(val) == 'None' else str(val)) for par, val in json.loads(args.model_params_for_tuning).items()}

        logger = logging.getLogger(' Data-driven modeling')
         
        logger.info(f' Starting data-driven modeling for steps id {args.id}')

        # Calling the data preparation pipeline
        data = data_preparation_pipeline(args)

        # Data
        X_train = data['X_train']
        Y_train = data['Y_train']
        X_test = data['X_test']
        Y_test = data['Y_test']
        del data

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
                    for row in list(range(i, input_parms.shape[0])):
                        worksheet.cell(row=row+4, column=col+1).value = input_val
                
                step_old = step_new


            vals = input_parms.iloc[i]
            args_dict = vars(args)
            args_dict.update({par: int(vals[idx+3]) if isnot_string(str(vals[idx+3])) else (None if str(vals[idx+3]) == 'None' else str(vals[idx+3])) for idx, par in enumerate(agrs_list)})
            if params['model'][args.data_driven_model]['model_params']['defined']:
                model_params = params['model'][args.data_driven_model]['model_params']['defined']
            else:
                model_params = params['model'][args.data_driven_model]['model_params']['default']
            model_params = {par: int(val) if isnot_string(str(val)) else (None if str(val) == 'None' else str(val)) for par, val in model_params.items()}
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
                modeling_results, classifier = modeling_pipeline(X_train, Y_train, 
                                            args.data_driven_model,
                                            args.model_params,
                                            return_model=True)


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

    # Tuning parameters for select model
    if "False" in args.model_params_for_tuning.keys():
        pass
    else:
        pass

    # Fitting the selected model with params
    modeling_results, classifier = modeling_pipeline(X_train, Y_train,
                                args.data_driven_model,
                                args.model_params,
                                return_model=True)

    # Evaluating the selected model
    error = prediction_evaluation(classifier, X_test, Y_test, metric='error')
    print(f'The {args.data_driven_model} model error: {error}')
    for key, val in modeling_results.items():
        print(f'The {args.data_driven_model} model {key.replace("_", " ")}: {val}') 

    # Persisting the selected model




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
    parser.add_argument('--intput_file',
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
                        default='{"False": "False"}')
    

    args = parser.parse_args()

    machine_learning_pipeline(args)