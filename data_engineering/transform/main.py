#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.transform.npi_transformer import transforming_npi
from data_engineering.transform.npri_transformer import transforming_npri
from data_engineering.transform.tri_transformer import transforming_tri
from data_engineering.transform.chemical_standardizing import normalizing_chemicals
from data_engineering.transform.industry_sector_standardizing import normalizing_sectors
from data_engineering.transform.database_normalization import database_normalization

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

    logger.info(' Running TRI transformer')
    transforming_tri()

    logger.info(' Running chemical standardizing')
    normalizing_chemicals()

    logger.info(' Running industry sector standardizing')
    normalizing_sectors()

    logger.info(' Running database normalization')
    database_normalization()


if __name__ == '__main__':

    tramsform_pipeline()
