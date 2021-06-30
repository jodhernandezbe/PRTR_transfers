#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is a Python script written for scraping the website that stored the PubChem data.
'''

# Importing libraries
from data_engineering.extract.common import config

import requests


def processing_json(response):
    '''
    Function to process response from the HTTP requests
    '''

    result = response.json()
    if (not result) or ('PropertyTable' not in result.keys()):
        infomation = None
    else:
        structure = result['PropertyTable']['Properties'][0]
        if 'CanonicalSMILES' not in structure.keys():
            infomation = None
        else:
            infomation = structure['CanonicalSMILES']

    return infomation


def looking_for_structure_details(cas_number):
    '''
    Function to obtain chemical SMILES from the PubChem
    '''

    # Calling configuration
    _config = config()['system']['PubChem']
    url = _config['url']
    rn_query_string = _config['by_rn'].format(cas_number=cas_number)
    registry_id_query_string = _config['by_registry_id'].format(cas_number=cas_number)

    # HTTP request
    try:
        response = requests.get(f'{url}/{rn_query_string}')
        if response.status_code == 200:
            return processing_json(response)
        elif response.status_code == 404:
            response = requests.get(f'{url}/{registry_id_query_string}')
            if response.status_code == 200:
                return processing_json(response)
            else:
                raise ValueError(f'Error: {response.status_code}')
        else:
            raise ValueError(f'Error: {response.status_code}')
    except ValueError as ve:
        print(f'{ve} for chemical {cas_number} (PubChem database)')
        return None