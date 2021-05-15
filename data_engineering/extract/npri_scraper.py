#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from common import config

import requests
import os
import lxml.html as html
import re

dir_path = os.path.dirname(os.path.realpath(__file__))
regex = re.compile(r'.*\/(NPRI.*)\.csv')
files = {'NPRI-INRP_DisposalsEliminations_1993-present': 'NPRI_disposals',
         'NPRI-INRP_DisposalsEliminations_TransfersTransferts_1993-present': 'NPRI_transfers'}


def download_npri():
    '''
    Function to download the NPRI transfers file
    '''

    _config = config()['system']['NPRI']
    url = _config['url']
    queries = _config['queries']
    tables = queries['tables']

    try:
        response = requests.get(url)
        if response.status_code == 200:
            home = response.content.decode('utf-8')
            parser = html.fromstring(home)
            links_to_tables = parser.xpath(tables)
            for link in links_to_tables:
                filename = re.search(regex, link).group(1)
                filenema_output = files[filename]
                r_file = requests.get(link, allow_redirects=True)
                open(f'{dir_path}/output/{filenema_output}.csv', 'wb').write(r_file.content)
        else:
            raise ValueError(f'Error: {response.status_code}')
    except ValueError as ve:
        print(ve)


if __name__ == '__main__':
    download_npri()
