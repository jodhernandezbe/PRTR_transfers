#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

Base = declarative_base()


def create_engine_session(password,
                        rdbms='mysql',
                        username='root',
                        host='127.0.0.1',
                        port='3306',
                        db_name='PRTR_transfers'):
    '''
    Function to create an engine and session in the RDBMS by SQLAlchemy
    '''

    url = f'{rdbms}://{username}:{password}@{host}:{port}/{db_name}'

    if database_exists(url):
        Engine = create_engine(url)
    else:
        create_database(url)
        Engine = create_engine(url)

    Session = sessionmaker(bind=Engine)

    return Engine, Session
