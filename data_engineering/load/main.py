#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from industry_sector import NationalToGenericCode, IsicToGenericCode
from base import Base, create_engine_session

import pandas as pd
import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path
logger = logging.getLogger(__name__)


def main(args):

    filename = args.filename
    password = args.password
    username = args.username
    host = args.host
    port = args.port
    db_name = args.db_name
    sql_file = args.sql_file

    Engine, Session = create_engine_session(password,
                            username=username,
                            host=host,
                            port=port,
                            db_name=db_name)

    if filename == 'isic_to_generic':
        Object = IsicToGenericCode
    elif filename == 'national_to_generic':
        Object = NationalToGenericCode

    Object.__table__.create(Engine, checkfirst=True)
    session = Session()

    path = f'{dir_path}/../transform/output/{filename}.csv'
    df = pd.read_csv(path)

    for idx, row in df.iterrows():
        context = row.to_dict()
        logger.info(f'Loading record {idx} from {filename} into DB')
        instance = Object(**context)
        session.add(instance)

    session.commit()
    session.close()

    if sql_file == 'True':
        os.system(f'mysqldump -u {username} -p{password} {db_name} > {dir_path}/output/{db_name}.sql')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('filename',
                        help='The file you want to load into the db',
                        type=str)
    parser.add_argument('--password',
                        help='The password for using MySQL RDBMS',
                        type=str)
    parser.add_argument('--username',
                        help='The username for using MySQL RDBMS',
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
                        default='PRTR_transfers_project')
    parser.add_argument('--sql_file',
                        help='Would you like to obtain .SQL file',
                        type=str,
                        default='False')

    args = parser.parse_args()

    main(args)
