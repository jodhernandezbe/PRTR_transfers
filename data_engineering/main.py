#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.extract.main import scraper_pipeline
from data_engineering.transform.main import tramsform_pipeline

import logging
logging.basicConfig(level=logging.INFO)


def data_engineering_pipeline():
    '''
    Function for creating the data engineering pipeline
    '''
    logger = logging.getLogger(' Data engineering')

    logger.info(' Starting data engineering')

    scraper_pipeline()
    tramsform_pipeline()

if __name__ == '__main__':

    data_engineering_pipeline()