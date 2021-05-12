#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from base import Base

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime


class NationalToGenericCode(Base):
    __tablename__ = 'national_to_generic_codes'

    national_generic_id = Column(Integer(), primary_key=True)
    national_code = Column(Integer(), nullable=False)
    national_name = Column(String(200), nullable=False)
    generic_sector_code = Column(Integer(), nullable=False)
    note = Column(String(10), nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    def __init__(self, **kwargs):
        self.national_generic_id = kwargs['national_generic_id']
        self.national_code = kwargs['national_code']
        self.national_name = kwargs['national_name']
        self.generic_sector_code = kwargs['generic_sector_code']
        self.note = kwargs['note']


class IsicToGenericCode(Base):
    __tablename__ = 'isic_to_generic_codes'

    isic_generic_id = Column(Integer(), primary_key=True)
    isic_code = Column(Integer(), nullable=False)
    isic_name = Column(String(200), nullable=False)
    generic_sector_code = Column(Integer(), nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    def __init__(self, **kwargs):
        self.isic_generic_id = kwargs['isic_generic_id']
        self.isic_code = kwargs['isic_code']
        self.isic_name = kwargs['isic_name']
        self.generic_sector_code = kwargs['generic_sector_code']
