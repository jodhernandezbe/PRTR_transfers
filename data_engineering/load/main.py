#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import argparse
import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import os

from industry_sector import NationalToGenericCode, IsicToGenericCode
from base import Base, Engine, Session

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path
logger = logging.getLogger(__name__)

def main(filename):
    Base.metadata.create_all(Engine)
    session = Session()

    if filename == 'isic_to_generic':
        Object = IsicToGenericCode
    elif filename == 'national_to_generic':
        Object = NationalToGenericCode

    path = f'{dir_path}/../transform/output/{filename}.csv'
    df = pd.read_csv(path)

    for idx, row in df.iterrows():
        context = row.to_dict()
        logger.info(f'Loading record {idx} from {filename} into DB')
        instance = Object(**context)
        session.add(instance)

    session.commit()
    session.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filename',
                        help='The file you want to load into the db',
                        type=str)

    args = parser.parse_args()

    main(args.filename)
