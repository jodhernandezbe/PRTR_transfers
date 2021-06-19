#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.load.base import Base

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey


class NationalGenericSubstance(Base):
    __tablename__ = 'national_generic_substance'

    national_generic_substance_id = Column(Integer(), primary_key=True)
    national_substance_prtr_system_id = Column(Integer(),
                                               ForeignKey('national_substance.national_substance_prtr_system_id'),
                                               nullable=False)
    generic_substance_id = Column(String(20),
                                ForeignKey('generic_substance.generic_substance_id'),
                                nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    def __init__(self, **kwargs):
        self.national_generic_substance_id = kwargs['national_generic_substance_id']
        self.national_substance_prtr_system_id = kwargs['national_substance_prtr_system_id']
        self.generic_substance_id = kwargs['generic_substance_id']


class NationalSubstance(Base):
    __tablename__ = 'national_substance'

    national_substance_prtr_system_id = Column(Integer(), primary_key=True)
    national_substance_id = Column(String(20), nullable=False)
    national_substance_name = Column(String(150), nullable=False)
    prtr_system = Column(String(5), nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    def __init__(self, **kwargs):
        self.national_substance_prtr_system_id = kwargs['national_substance_prtr_system_id']
        self.national_substance_id = kwargs['national_substance_id']
        self.national_substance_name = kwargs['national_substance_name']
        self.prtr_system = kwargs['prtr_system']


class GenericSubstance(Base):
    __tablename__ = 'generic_substance'

    generic_substance_id = Column(String(20), primary_key=True)
    generic_substance_name = Column(String(300), nullable=False)
    cas_number = Column(String(20), nullable=True)
    created_at = Column(DateTime(), default=datetime.now())

    def __init__(self, **kwargs):
        self.generic_substance_id = kwargs['generic_substance_id']
        self.generic_substance_name = kwargs['generic_substance_name']
        self.cas_number = kwargs['cas_number']