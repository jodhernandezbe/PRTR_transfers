#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is a python script which uses descriptors from the RDKit, an open-source cheminformatics software (https://www.rdkit.org/)

List of RDKit available descriptors: https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
'''

# Importing libraries
from data_engineering.extract.nlm_scraper import looking_for_structure_details

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path

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
        descriptors = {descriptor_name: [descriptor_func(molecule)] for
                       descriptor_name, descriptor_func
                       in rdkit_descriptors().items()}

    return descriptors


def information_for_set_of_chems(
                                 col_name,
                                 col_smiles,
                                 col_id,
                                 ):
    '''
    This is a function to look for the descriptors for all molecules
    '''

    # Looking for chemical structure description notations (SMILE, InChi, InChiKey)
    

    # Iterating over the dataframe rows (chemicals)
    df_descriptors = pd.DataFrame()
    for idx, row in df_chems.iterrows():
        descriptors = descriptors_for_chemical(row[col_smiles])
        if descriptors is None:
            print('Descriptors for {name}'
                .format(name=row[col_name]))
        else:
            descriptors.update({'CASN': row[col_id]})
            df_descriptors = \
                pd.concat([df_descriptors,
                           pd.DataFrame(descriptors)])
            del descriptors

    # Changing the names of the columns in df_chem
    df_chems.rename(columns={col_name: 'Name',
                            col_smiles: 'SMILES',
                            col_id: 'CASN'},
                    inplace=True)


    # Merging descriptors and input parameters
    df_descriptors = pd.merge(df_descriptors,
                              df_chems,
                              how='inner',
                              on='CASN')
    del df_chems

    # Dropping columns with same value for all records
    df_descriptors.drop(df_descriptors.std()[(df_descriptors.std() == 0)].index,
                        axis=1, inplace=True)


    # Saving the information
    df_descriptors.to_csv(output_path, sep=',', index=False)


if __name__ == "__main__":
    pass