#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import yaml
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

__config = None


def config():
    '''
    Function to load the configuration file for scraping each PRTR website
    '''

    global __config
    if not __config:
        with open(f'{dir_path}/config.yaml',
                  mode='r') as f:
            __config = yaml.load(f, Loader=yaml.FullLoader)
    return __config
