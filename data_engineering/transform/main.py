#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.transform.npi_transformer import transforming_npi

import logging
logging.basicConfig(level=logging.INFO)

def tramsform_pipeline():
    '''
    Function for creating the transform pipeline for the PRTR systems
    '''

    logger = logging.getLogger(' Data engineering --> Extract')

    logger.info(' Running NPI scraper')
    transforming_npi()



if __name__ == '__main__':

    tramsform_pipeline()