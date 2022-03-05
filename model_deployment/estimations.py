#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from rdkit import Chem
from rdkit.Chem import Descriptors

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


def get_estimations(input_features_dict, prob: bool = False, transfer_class='mlc'):
    '''
    Function to get the estimates for the input features
    '''

    if transfer_class == 'mlc':
        pass
    else:
        pass

    

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
                descriptors.update({descriptor_name: [descriptor_func(molecule)]})
            except ZeroDivisionError:
                descriptors.update({descriptor_name: None})

    return descriptors