#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.load.industry_sector import NationalGenericSector, NationalSector, GenericSector
from data_engineering.load.facility import Facility
from data_engineering.load.prtr_system import PRTRSystem
from data_engineering.load.record import TransferRecord
from data_engineering.load.substance import NationalGenericSubstance, NationalSubstance, GenericSubstance
from data_engineering.load.transfer import NationalGenericTransferClass, NationalTransferClass, GenericTransferClass
from data_engineering.load.chemical import GenericSubstanceChemicalInCategory, ChemicalInCategory
from data_engineering.load.base import Base, create_engine_session

import pandas as pd
import os
import zipfile
import argparse
import logging
logging.basicConfig(level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path

# Dictionary to associate each table file with each table in the SQL database
Dic_tables = {'generic_sector': GenericSector,
              'generic_substance': GenericSubstance,
              'generic_transfer_class': GenericTransferClass,
              'national_sector': NationalSector,
              'national_substance': NationalSubstance,
              'national_transfer_class': NationalTransferClass,
              'chemical_in_category': ChemicalInCategory, 
              'national_generic_sector': NationalGenericSector,
              'national_generic_substance': NationalGenericSubstance,
              'national_generic_transfer_class': NationalGenericTransferClass,
              'generic_substance_chemical_in_category': GenericSubstanceChemicalInCategory,
              'facility': Facility,
              'prtr_system': PRTRSystem,
              'transfer_record': TransferRecord}


def function_to_create_zip(filename):
    '''
    Function to create a zip file for .sql files
    '''

    with zipfile.ZipFile(f'{dir_path}/output/{filename}.zip', 'w') as zipObj2:
        zipObj2.write(f'{dir_path}/output/{filename}.sql',
                    os.path.basename(f'{dir_path}/output/{filename}.sql'),
                    compress_type=zipfile.ZIP_DEFLATED)


def load_pipeline(args):

    logger = logging.getLogger(' Data engineering --> Load')

    password = args.password
    rdbms = args.rdbms
    username = args.username
    host = args.host
    port = args.port
    db_name = args.db_name
    sql_file = args.sql_file

    Engine, Session = create_engine_session(password,
                            rdbms=rdbms,
                            username=username,
                            host=host,
                            port=port,
                            db_name=db_name)

    for filename in reversed(list(Dic_tables.keys())):
        Object = Dic_tables[filename]
        Object.__table__.drop(Engine, checkfirst=True)

    # Saving each table
    for filename, Object in Dic_tables.items():
        Object.__table__.create(Engine, checkfirst=True)
        session = Session()
        path = f'{dir_path}/../transform/output/{filename}.csv'
        df = pd.read_csv(path, encoding='utf-8')
        df = df.where((pd.notnull(df)), None)
        logger.info(f' Loading table {filename} into the {db_name} database')
        # Saving each record by table
        for _, row in df.iterrows():
            context = row.to_dict()
            instance = Object(**context)
            session.add(instance)
        session.commit()
        session.close()

    if sql_file == 'True':
        if rdbms == 'mysql':
            exporting_string = f'mysqldump -u {username} -p{password} {db_name} > {dir_path}/output/{db_name}_v_MySQL.sql'
            function_to_create_zip(f'{db_name}_v_MySQL')
        elif rdbms == 'postgresql':
            exporting_string = f'pg_dump -U {username} -h {host} {db_name} -W > {dir_path}/output/{db_name}_v_PostgreSQL.sql'
            function_to_create_zip(f'{db_name}_v_PostgreSQL')

        os.system(exporting_string)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rdbms',
                        help='The Relational Database Management System (RDBMS) you would like to use',
                        choices=['mysql', 'postgresql'],
                        type=str,
                        default='mysql')
    parser.add_argument('--password',
                        help='The password for using the RDBMS',
                        type=str)
    parser.add_argument('--username',
                        help='The username for using the RDBMS',
                        type=str,
                        default='root')
    parser.add_argument('--host',
                        help='The computer hosting for the database',
                        type=str,
                        default='127.0.0.1')
    parser.add_argument('--port',
                        help='Port used by the database engine',
                        type=str,
                        default='3306')
    parser.add_argument('--db_name',
                        help='Database name',
                        type=str,
                        default='PRTR_transfers')
    parser.add_argument('--sql_file',
                        help='Would you like to obtain .SQL file',
                        choices=['True', 'False'],
                        type=str,
                        default='False')

    args = parser.parse_args()

    load_pipeline(args)
