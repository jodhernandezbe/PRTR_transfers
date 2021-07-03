#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.load.base import Base

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship


class Facility(Base):
    __tablename__ = 'facility'

    national_facility_and_generic_sector_id = Column(Integer(), primary_key=True)
    national_facility_id = Column(String(20), nullable=False)
    national_generic_sector_id = Column(Integer(),
                                        ForeignKey('national_generic_sector.national_generic_sector_id',
                                                ondelete='CASCADE',
                                                onupdate='cascade'),
                                        nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    national_generic_sector = relationship("NationalGenericSector",
                                        back_populates="facility")
    transfer_record = relationship("TransferRecord",
                                    back_populates="facility")

    def __init__(self, **kwargs):
        self.national_facility_and_generic_sector_id = kwargs['national_facility_and_generic_sector_id']
        self.national_facility_id = kwargs['national_facility_id']
        self.national_generic_sector_id = kwargs['national_generic_sector_id']