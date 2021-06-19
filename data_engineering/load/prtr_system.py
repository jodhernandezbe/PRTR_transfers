#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.load.base import Base

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Integer


class PRTRSystem(Base):
    __tablename__ = 'prtr_system'

    id = Column(Integer(), primary_key=True, autoincrement=True)
    prtr_system = Column(String(5), nullable=False)
    country = Column(String(3), nullable=False)
    industry_classification_system = Column(String(10), nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    def __init__(self, **kwargs):
        self.prtr_system = kwargs['prtr_system']
        self.country = kwargs['country']
        self.industry_classification_system = kwargs['industry_classification_system']