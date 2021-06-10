#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is a Python script written for scraping the website that stored the Substance
Registry Services (SRS) data. The SRS is the U.S. Environmental Protection Agency’s
(EPA) central registry for information about regulated and monitored substances. 
'''

# Importing libraries
from data_engineering.extract.common import config

import requests

def get_cas_by_alternative_id(altId='N230',
                            altIdType='22',
                            substanceName='Certain glycol ethers'):
    '''
    Function to get the CAS number by substance alternative IDs
    '''

    # Calling configuration
    _config = config()['system']['SRS']
    url = _config['url']
    id_query_string = _config['by_alternative_id'].format(altId=altId, altIdType=altIdType)
    name_query_string = _config['by_name'].format(substanceName=substanceName)

    # HTTP request
    try:
        response = requests.get(f'{url}/{id_query_string}')
        if response.status_code == 200:
            json = response.json()
            if not json:
                result = get_cas_by_name(**{'name_query_string': name_query_string})
            else:
                result = json[0]['currentCasNumber']
            return result
        else:
            raise ValueError(f'Error: {response.status_code}')
    except ValueError as ve:
        print(ve)


def get_cas_by_name(**kwargs):
    '''
    Function to get CAS number by substance name
    '''

    # Checking inputs
    _config = config()['system']['SRS']
    url = _config['url']
    if 'name_query_string' in kwargs.keys():
        name_query_string = kwargs['name_query_string']
    else:
        name_query_string = _config['by_name'].format(**kwargs)
    
    # HTTP request
    try:
        response = requests.get(f'{url}/{name_query_string}')
        if response.status_code == 200:
            json = response.json()
            if not json:
                result = None
            else:
                result = json[0]['currentCasNumber']
            return result
        else:
            raise ValueError(f'Error: {response.status_code}')
    except ValueError as ve:
        print(ve)


def get_generic_name_by_cas(casNum='1336-36-3'):
    '''
    Function to get name by cas number
    '''
    
    # Calling configuration
    _config = config()['system']['SRS']
    url = _config['url']
    cas_query_string = _config['by_cas'].format(casNum=casNum)

    # HTTP request
    try:
        response = requests.get(f'{url}/{cas_query_string}')
        if response.status_code == 200:
            json = response.json()
            if not json:
                result = None
            else:
                result = json[0]['systematicName']
            return result
        else:
            raise ValueError(f'Error: {response.status_code}')
    except ValueError as ve:
        print(ve)