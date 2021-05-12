#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()

def create_engine_session(password,
                        username='root',
                        host='127.0.0.1',
                        port='3306',
                        db_name='PRTR_transfers_project'):
    '''
    Function to create an engine and session in MySQL RDBMS by SQLAlchemy
    '''

    Engine = create_engine(f'mysql://{username}:{password}@{host}:{port}')
    Engine.execute(f'CREATE DATABASE IF NOT EXISTS {db_name}')
    Engine.execute(f'USE {db_name}')
    Session = sessionmaker(bind=Engine)

    return Engine, Session
