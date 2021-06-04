#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.transform.npi_transformer import transforming_npi
from data_engineering.transform.npri_transformer import transforming_npri

import logging
logging.basicConfig(level=logging.INFO)

def tramsform_pipeline():
    '''
    Function for creating the transform pipeline for the PRTR systems
    '''

    logger = logging.getLogger(' Data engineering --> Transform')

    logger.info(' Running NPI transformer')
    transforming_npi()
    
    logger.info(' Running NPRI transformer')
    transforming_npri()



if __name__ == '__main__':

    tramsform_pipeline()
