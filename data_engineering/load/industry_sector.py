#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.load.base import Base

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship


class NationalGenericSector(Base):
    __tablename__ = 'national_generic_sector'

    national_generic_sector_id = Column(Integer(), primary_key=True)
    national_sector_id = Column(Integer(),
                                ForeignKey('national_sector.national_sector_id',
                                        ondelete='CASCADE',
                                        onupdate='cascade'),
                                nullable=False)
    generic_sector_code = Column(Integer(),
                                ForeignKey('generic_sector.generic_sector_code',
                                        ondelete='CASCADE',
                                        onupdate='cascade'),
                                nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    national_sector = relationship("NationalSector",
                                    back_populates="national_generic_sector")
    generic_sector = relationship("GenericSector",
                                back_populates="national_generic_sector")
    facility = relationship("Facility",
                            back_populates="national_generic_sector")

    def __init__(self, **kwargs):
        self.national_generic_sector_id = kwargs['national_generic_sector_id']
        self.national_sector_id = kwargs['national_sector_id']
        self.generic_sector_code = kwargs['generic_sector_code']


class NationalSector(Base):
    __tablename__ = 'national_sector'

    national_sector_id = Column(Integer(), primary_key=True)
    national_sector_code = Column(Integer(), nullable=False)
    national_sector_name = Column(String(250), nullable=False)
    industry_classification_system = Column(String(10), nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    national_generic_sector = relationship("NationalGenericSector",
                                        back_populates="national_sector",
                                        uselist=False)

    def __init__(self, **kwargs):
        self.national_sector_id = kwargs['national_sector_id']
        self.national_sector_code = kwargs['national_sector_code']
        self.national_sector_name = kwargs['national_sector_name']
        self.industry_classification_system = kwargs['industry_classification_system']


class GenericSector(Base):
    __tablename__ = 'generic_sector'

    generic_sector_code = Column(Integer(), primary_key=True)
    generic_sector_name = Column(String(250), nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    national_generic_sector = relationship("NationalGenericSector",
                                        back_populates="generic_sector")

    def __init__(self, **kwargs):
        self.generic_sector_code = kwargs['generic_sector_code']
        self.generic_sector_name = kwargs['generic_sector_name']
