#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.load.base import Base

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship


class GenericSubstanceChemicalInCategory(Base):
    __tablename__ = 'generic_substance_chemical_in_category'

    generic_substance_chemical_in_category_id = Column(Integer(), primary_key=True)
    chemical_in_category_cas = Column(String(20),
                                    ForeignKey('chemical_in_category.chemical_in_category_cas',
                                            ondelete='CASCADE',
                                            onupdate='cascade'),
                                    nullable=False)
    generic_substance_id = Column(String(20),
                                ForeignKey('generic_substance.generic_substance_id',
                                        ondelete='CASCADE',
                                        onupdate='cascade'),
                                nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    chemical_in_category = relationship("ChemicalInCategory",
                                    back_populates="generic_substance_chemical_in_category")
    generic_substance = relationship("GenericSubstance",
                                    back_populates="generic_substance_chemical_in_category")

    def __init__(self, **kwargs):
        self.generic_substance_chemical_in_category_id = kwargs['generic_substance_chemical_in_category_id']
        self.chemical_in_category_cas = kwargs['chemical_in_category_cas']
        self.generic_substance_id = kwargs['generic_substance_id']


class ChemicalInCategory(Base):
    __tablename__ = 'chemical_in_category'

    chemical_in_category_cas = Column(String(20), primary_key=True)
    chemical_in_category_name = Column(String(300), nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    generic_substance_chemical_in_category = relationship("GenericSubstanceChemicalInCategory",
                                                        back_populates="chemical_in_category")

    def __init__(self, **kwargs):
        self.chemical_in_category_cas = kwargs['chemical_in_category_cas']
        self.chemical_in_category_name = kwargs['chemical_in_category_name']