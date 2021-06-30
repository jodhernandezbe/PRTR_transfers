#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is a python script which uses descriptors from the RDKit, an open-source cheminformatics software (https://www.rdkit.org/)

List of RDKit available descriptors: https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
'''

# Importing libraries
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

def rdkit_descriptors():
    '''
    This is a function for getting the list of all descriptos in RDKit package
    '''

    # List of attributes to drop
    Methods_exception = [
                         '_FingerprintDensity',
                         '_isCallable', '_runDoctests',
                         '_setupDescriptors',
                         'setupAUTOCorrDescriptors',
                         '_ChargeDescriptors'
                         ]

    # Getting list of attributes as functions
    methods =  {func: getattr(Descriptors, func) for func in dir(Descriptors)
                if type(getattr(Descriptors, func)).__name__ == "function"
                and func not in Methods_exception}
    methods = {s: methods[s] for s in sorted(methods)}

    return methods


def descriptors_for_chemical(SMILES):
    '''
    This is a function for collecting all the descriptor for a molecule
    '''

    descriptors = None

    # Molecule from SMILES
    molecule = Chem.MolFromSmiles(SMILES)

    if molecule is not None:
        # Molecular descriptors
        descriptors = {}
        for descriptor_name, descriptor_func in rdkit_descriptors().items():
            try:
                descriptors.update({descriptor_name: [descriptor_func(molecule)]})
            except ZeroDivisionError:
                descriptors.update({descriptor_name: None})

    return descriptors


def information_for_set_of_chems(
                                 col_id,
                                 df_chems
                                 ):
    '''
    This is a function to look for the descriptors for all molecules
    '''    

    # Iterating over the dataframe rows (chemicals)
    df_descriptors = pd.DataFrame()
    for _, row in df_chems.iterrows():
        if row['smiles']:
            descriptors = descriptors_for_chemical(row['smiles'])
            if descriptors is None:
                continue
            else:
                descriptors.update({col_id: row[col_id]})
                df_descriptors = \
                    pd.concat([df_descriptors,
                            pd.DataFrame(descriptors)])
                del descriptors
        else:
            continue

    # Merging descriptors and input parameters
    df_chems = pd.merge(df_descriptors,
                        df_chems,
                        how='right',
                        on=col_id)
    del df_descriptors

    return df_chems