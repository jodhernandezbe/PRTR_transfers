#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path
db_name = 'PRTR_transfers_project.db'

Engine = create_engine(f'sqlite:///{dir_path}/output/{db_name}')
Session = sessionmaker(bind=Engine)

Base = declarative_base()
