#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import pandas as pd
import os
import zipfile
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.sql import text
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, ProgrammingError

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path


def create_database_from_sql_file(Engine,
                                rdbms='mysql',
                                db_name='PRTR_transfers'):
    '''
    Function to create a database based on SQL file
    '''

    # Checking .sql file
    if rdbms == 'mysql':
        string = 'MySQL'
    else:
        string = 'PostgreSQL'
    with zipfile.ZipFile(f'{dir_path}/../data_engineering/load/output/{db_name}_v_{string}.zip', 'r') as zip_ref:
        zip_ref.extractall(f'{dir_path}/../data_engineering/load/output')
    
    # Parsing and creating the database from .sql file
    with Engine.connect() as connection:
        with open(f'{dir_path}/../data_engineering/load/output/{db_name}_v_{string}.sql', 'r') as sql_file:
            sql_as_string = sql_file.read()
            sqlcommands = sql_as_string.split(';')
            for command in sqlcommands:
                sql_command = text(command)
                try:
                    result = connection.execute(sql_command)
                except (ProgrammingError, OperationalError):
                    continue


def create_engine_instance(password,
                rdbms='mysql',
                username='root',
                host='127.0.0.1',
                port='3306',
                db_name='PRTR_transfers'):
    '''
    Function to create an SQL engine
    '''

    # URL string
    url = f'{rdbms}://{username}:{password}@{host}:{port}/{db_name}'
    if rdbms == 'mysql':
        url = f'{url}?charset=utf8mb4'

    if not database_exists(url):
        Is_created = True
        create_database(url)
    else:
        Is_created = False

    # Creating engine
    if rdbms == 'mysql':
        Engine = create_engine(url)
    else:
        Engine = create_engine(url, client_encoding='utf8')

    # Creating database if it's needed
    if Is_created:
        create_database_from_sql_file(Engine)

    return Engine


def opening_dataset(args):
    '''
    Function to open dataset based on sql query
    '''

    password = args.password
    rdbms = args.rdbms
    username = args.username
    host = args.host
    port = args.port
    db_name = args.db_name
    
    # Creating engine
    Engine = create_engine_instance(password,
                            rdbms=rdbms,
                            username=username,
                            host=host,
                            port=port,
                            db_name=db_name)

    # Query for selecting only generic substances that are groups of chemicals
    sql_query_chemicals = '''
    SELECT gs.generic_substance_id,
        gscic.chemical_in_category_cas
    FROM generic_substance AS gs
    INNER JOIN generic_substance_chemical_in_category as gscic
    ON gscic.generic_substance_id = gs.generic_substance_id;
    '''

    # Query for selecting only generic substances that are chemicals
    sql_query_subscantes = '''
    SELECT gs.generic_substance_id,
        gs.cas_number
    FROM generic_substance AS gs
    LEFT JOIN generic_substance_chemical_in_category as gscic
    ON gscic.generic_substance_id = gs.generic_substance_id
    WHERE gscic.generic_substance_id IS NULL;
    '''

    # Query for fetching transfer records
    sql_query_record = '''
    SELECT tr.transfer_record_id,
	   tr.reporting_year,
	   tr.transfer_amount_kg,
	   tr.reliability_score,
	   ngs.generic_substance_id,
	   gtc.generic_transfer_class_id,
	   gtc.transfer_class_wm_hierarchy_name,
	   tr.national_facility_and_generic_sector_id,
	   ngse.generic_sector_code
    FROM transfer_record AS tr
    INNER JOIN national_generic_substance AS ngs
    ON tr.national_generic_substance_id = ngs.national_generic_substance_id
    INNER JOIN national_generic_transfer_class AS ngtc
    ON ngtc.national_generic_transfer_class_id = tr.national_generic_transfer_class_id
    INNER JOIN generic_transfer_class AS gtc
    ON gtc.generic_transfer_class_id = ngtc.generic_transfer_class_id
    INNER JOIN facility AS f
    ON f.national_facility_and_generic_sector_id = tr.national_facility_and_generic_sector_id
    INNER JOIN national_generic_sector AS ngse
    ON f.national_generic_sector_id = ngse.national_generic_sector_id;
    '''

    sql_query_dict = {'chemical': sql_query_chemicals,
                      'substance': sql_query_subscantes,
                      'record': sql_query_record}

    # Fetching the PRTR information to build the data-driven models
    df_dict = {}
    for table, query in sql_query_dict.items():
        df = pd.read_sql_query(query, Engine)
        df_dict.update({table: df})
        del df

    return df_dict