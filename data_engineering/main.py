#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.extract.main import scraper_pipeline
from data_engineering.transform.main import tramsform_pipeline
from data_engineering.load.main import load_pipeline

import logging
import argparse
logging.basicConfig(level=logging.INFO)


def data_engineering_pipeline(args):
    '''
    Function for creating the data engineering pipeline
    '''
    logger = logging.getLogger(' Data engineering')

    logger.info(' Starting data engineering')

    # Calling web scraping pipeline
    scraper_pipeline()

    # Calling database transforming pipeline
    tramsform_pipeline()

    # Calling database loading pipeline
    load_pipeline(args)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rdbms',
                        help='The Relational Database Management System (RDBMS) you would like to use',
                        choices=['mysql', 'postgresql'],
                        type=str,
                        default='mysql')
    parser.add_argument('--password',
                        help='The password for using the RDBMS',
                        type=str)
    parser.add_argument('--username',
                        help='The username for using the RDBMS',
                        type=str,
                        default='root')
    parser.add_argument('--host',
                        help='The computer hosting for the database',
                        type=str,
                        default='127.0.0.1')
    parser.add_argument('--port',
                        help='Port used by the database engine',
                        type=str,
                        default='3306')
    parser.add_argument('--db_name',
                        help='Database name',
                        type=str,
                        default='PRTR_transfers')
    parser.add_argument('--sql_file',
                        help='Would you like to obtain .SQL file',
                        choices=['True', 'False'],
                        type=str,
                        default='False')

    args = parser.parse_args()

    data_engineering_pipeline(args)