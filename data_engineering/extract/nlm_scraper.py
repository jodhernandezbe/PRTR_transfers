#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is a Python script written for scraping the website that stored the National Library of Medicine (NLM) data.
The NLM is part of the National Institutes of Health (NIH), U.S. Department of Health and Human Services.
'''
# Importing libraries
from data_engineering.extract.common import config

import requests

def looking_for_structure_details(cas_number):
    '''
    Function to obtain chemical SMILES from the U.S. NLM
    '''

    # Calling configuration
    _config = config()['system']['NLM']
    url = _config['url']
    rn_query_string = _config['by_register_number'].format(cas_number=cas_number)

    # HTTP request
    try:
        response = requests.get(f'{url}/{rn_query_string}')
        if response.status_code == 200:
            result = response.json()['results'][0]
            if (not result) or ('structureDetails' not in result.keys()):
                infomation = None
            else:
                structure = result['structureDetails']
                if 's' not in structure.keys():
                    infomation = None
                else:
                    infomation = structure['s']
            return infomation
        else:
            raise ValueError(f'Error: {response.status_code}')
    except ValueError as ve:
        print(f'{ve} for chemical {cas_number} (NLM database)')
        return None