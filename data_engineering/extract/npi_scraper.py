#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is a Python script written for scraping the website that stored the National
Pollutant Inventory (NPI) data. The NPI is the Australian Pollutant Release and
Transfer Register (PRTR)
'''

# Importing libraries
from common import config

import requests
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


def download_npi():
    '''
    Function to download the NPI transfers file
    '''

    _config = config()['system']['NPI']
    url = _config['url']
    resource_id = _config['resource_id']

    for key, id in resource_id.items():

        try:
            response = requests.get(f'{url}?sql=SELECT * from "{id}"')
            if response.status_code == 200:
                json = response.json()
                result = json['result']
                records = result['records']
                df = pd.DataFrame(records)
                df.to_csv(f'{dir_path}/output/NPI_{key}.csv',
                        index=False)
            else:
                raise ValueError(f'Error: {response.status_code}')
        except ValueError as ve:
            print(ve)


if __name__ == '__main__':
    download_npi()
