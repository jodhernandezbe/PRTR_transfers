#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import pandas as pd
import numpy as np
import ast

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path
processor_path = os.path.join(dir_path, 'processor') # processor path
classifier_path = os.path.join(processor_path, 'classifiers') # classifier path
dict_to_process = {'mlc': 6,
                   'M1': 8,
                   'M2': 11,
                   'M3': 14,
                   'M4': 17,
                   'M5': 20,
                   'M6': 23,
                   'M7': 26,
                   'M8': 29,
                   'M9': 32,
                   'M10': 35}

def interval_type(string_interval, flow):
    """
    Parse interval string to Interval

    Taken from: https://stackoverflow.com/questions/65295837/turn-string-representation-of-interval-into-actual-interval-in-pandas 
    """
    
    table = str.maketrans({'[': '(', ']': ')'})
    left_closed = string_interval.startswith('[')
    right_closed = string_interval.endswith(']')

    left, right = ast.literal_eval(string_interval.translate(table))

    t = 'neither'
    if left_closed and right_closed:
        t = 'both'
    elif left_closed:
        t = 'left'
    elif right_closed:
        t = 'right'

    return pd.IntervalIndex.from_breaks([left, right], closed=t).contains(flow)[0]


def organizing_features(input_features_dict, id_number):
    '''
    Function to organize input features for model
    '''
    
    # Calling feature order
    feature_order = pd.read_csv(os.path.join(processor_path,
                                                f'input_features_id_{id_number}.csv'),
                                    header=None
                                    )

    # Chemical descriptors
    chem_descriptors = feature_order.loc[~feature_order[1].isin(input_features_dict.keys()), 1].tolist()
    descriptors = descriptors_for_chemical(input_features_dict['smiles'], chem_descriptors)
    if descriptors:
        input_features_dict.update(descriptors)
        del input_features_dict['smiles']

        # Checking flow
        flow_intervals = pd.read_csv(os.path.join(processor_path,
                                    f'flow_handling_discretizer_id_{id_number}.csv'))
        flow_bin = flow_intervals.loc[flow_intervals['Flow rate interval [kg]'].apply(lambda x: interval_type(x, input_features_dict['transfer_amount_kg'])), 'Value'].values[0]
        input_features_dict.update({'transfer_amount_kg': flow_bin})

        # Industry sector
        sector_encoding = pd.read_csv(os.path.join(processor_path, f'generic_sector_for_params_id_{id_number}.csv'))
        encoding = sector_encoding[sector_encoding[f'sector_{input_features_dict["generic_sector_code"]}'] == 1].to_dict(orient='record')[0]
        input_features_dict.update(encoding)
        del input_features_dict["generic_sector_code"]

        # Scaling features
        input_features_dict.update({'transfer_amount_kg': flow_bin/10})
        df_scaling = pd.read_csv(os.path.join(processor_path, f'input_features_scaling_id_{id_number}.csv'),
                                    index_col='feature')
        for idx, row in df_scaling.iterrows():
            if idx in input_features_dict.keys():
                input_features_dict.update({idx: (input_features_dict[idx] - row['min'] ) / (row['max'] - row['min'])})
            else:
                continue

        # Organizing
        order_features = [val for key, val in input_features_dict.items() if ('transfer' in key) or ('sector' in key) or ('epsi' in key) or ('gva' in key) or ('price_usd_g' in key)]
        order_features = order_features + [val for key, val in input_features_dict.items() if ('transfer' not in key) and ('sector' not in key) and ('epsi' not in key) and ('gva' not in key) and ('price_usd_g' not in key)]

        return np.array(order_features)

    else:
        return None

def get_estimations(input_features_dict, prob: bool = False, transfer_class='mlc'):
    '''
    Function to get the estimates for the input features
    '''

    if transfer_class == 'mlc':

        id_number = dict_to_process['mlc']

        # Processing input features
        processed_features = organizing_features(input_features_dict, id_number)

    else:

        for t_class in transfer_class:
            
            id_number = dict_to_process[t_class]

            # Processing input features
            processed_features = organizing_features(input_features_dict, id_number)

    

def rdkit_descriptors(methods_to_keep):
    '''
    This is a function for getting the list of all molecular descriptors in RDKit package
    '''

    # Getting list of attributes as functions
    methods =  {func: getattr(Descriptors, func) for func in dir(Descriptors)
                if type(getattr(Descriptors, func)).__name__ == "function"
                and func in methods_to_keep}
    methods = {s: methods[s] for s in sorted(methods)}

    return methods


def descriptors_for_chemical(SMILES, methods_to_keep):
    '''
    This is a function for collecting all the descriptor for a molecule
    '''

    descriptors = None

    # Molecule from SMILES
    molecule = Chem.MolFromSmiles(SMILES)

    if molecule is not None:
        # Molecular descriptors
        descriptors = {}
        for descriptor_name, descriptor_func in rdkit_descriptors(methods_to_keep).items():
            try:
                descriptors.update({descriptor_name: descriptor_func(molecule)})
            except ZeroDivisionError:
                descriptors.update({descriptor_name: None})

    return descriptors