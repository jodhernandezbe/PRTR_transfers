#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from npi_scraper import download_npi
from npri_scraper import download_npri
from tri_scraper import TRI_Scrapper

import logging
logging.basicConfig(level=logging.INFO)


def scraper_pipeline():
    '''
    Function for creating the web scraping pipeline for the PRTR systems
    '''

    logger = logging.getLogger(' Data engineering --> Extract')

    logger.info(' Running NPI scraper')
    download_npi()

    logger.info(' Running NPRI scraper')
    download_npri()

    logger.info(' Running TRI scraper')
    Scrapper = TRI_Scrapper()
    Scrapper.extacting_tri_data_files(['3a', '3b', '3c'])


if __name__ == '__main__':

    scraper_pipeline()