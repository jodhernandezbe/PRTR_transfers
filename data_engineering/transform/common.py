#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import yaml
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

file = None


def config(filepath):
    '''
    Function to load yaml files with information needed for transforming the data sources
    '''

    global file
    if not file:
        with open(filepath,
                  mode='r') as f:
            file = yaml.load(f, Loader=yaml.FullLoader)
    return file
