#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This is a Python script written for scraping the website that stored the Toxics Release Inventory (TRI)
data. The TRI is the US Pollutant Release and Transfer Register (PRTR)
'''

# Importing libraries
from data_engineering.extract.common import config

import os
import requests
import lxml.html as html
import re
import pandas as pd
import zipfile
import io
import csv

class TRI_Scrapper:

    def __init__(self):
        self._dir_path = os.path.dirname(os.path.realpath(__file__))
        self._config = config()['system']['TRI']
        self._queries = self._config['queries']
        self._url = self._config['url']


    def _visit(self):
        '''
        Method for visiting the TRI website that stores the historic record data
        '''

        regex = re.compile(r'https://.*/US_([0-9]{4}).zip')

        try:
            response = requests.get(self._url)
            if response.status_code == 200:
                home = response.content.decode('utf-8')
                parser = html.fromstring(home)
                links = parser.xpath(self._queries['options'])
                zip_urls = dict()
                for link in links:
                   year = re.search(regex, link).group(1)
                   zip_urls.update({year: link})
                return zip_urls
            else:
                raise ValueError(f'Error: {response.status_code}')
        except ValueError as ve:
            print(ve)
    

    def _calling_tri_columns(self, key):
        '''
        Method for calling column positions and names for TRI files 
        '''

        path_columns = f'{self._dir_path}/../../ancillary'
        columns = pd.read_csv(f'{path_columns}/TRI_File_{key}_columns.txt',
                              header=None)
        columns = columns[0].tolist()
        return columns            


    def extacting_tri_data_files(self, keys):
        '''
        Method for extracting information for each TRI file by year
        '''

        # Calling the file sorted column names
        colum_names = dict()
        for key in keys:
            colum_names.update({key: self._calling_tri_columns(key)})

        # Unzipping and organizing the TRI files
        zip_urls = self._visit()
        for year, zip_url in zip_urls.items():
            zip_file = requests.get(zip_url)
            for key in keys:
                if (key == '3a') or ((key == '3b') and (int(year) <= 2010)) or ((key == '3c') and (int(year) >= 2011)):
                    with zipfile.ZipFile(io.BytesIO(zip_file.content)) as z_file:
                        z_file.extract(f'US_{key}_{year}.txt' ,
                                    f'{self._dir_path}/output')
                    df = pd.read_csv(f'{self._dir_path}/output/US_{key}_{year}.txt',
                                    header=None, encoding='ISO-8859-1',
                                    error_bad_lines=False,
                                    sep='\t', low_memory=True,
                                    skiprows=[0], engine='python',
                                    #lineterminator='\n',
                                    usecols=range(len(colum_names[key])),
                                    quoting=csv.QUOTE_NONE
                                    )
                    df.columns = colum_names[key]
                    df.to_csv(f'{self._dir_path}/output/US_{key}_{year}.csv',
                            sep=',', index=False)
                    existing = False
                    while not existing:
                        existing = os.path.exists(f'{self._dir_path}/output/US_{key}_{year}.csv')
                    os.remove(f'{self._dir_path}/output/US_{key}_{year}.txt')
        

if __name__ == '__main__':


    Scrapper = TRI_Scrapper()
    Scrapper.extacting_tri_data_files(['3a', '3b', '3c'])