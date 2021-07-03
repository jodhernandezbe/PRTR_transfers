#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.load.base import Base

from datetime import datetime
from sqlalchemy import Column, Integer, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship


class TransferRecord(Base):
    __tablename__ = 'transfer_record'

    transfer_record_id = Column(Integer(), primary_key=True)
    reporting_year = Column(Integer(), nullable=False)
    national_generic_substance_id = national_generic_substance_id = Column(Integer(),
                                                                            ForeignKey('national_generic_substance.national_generic_substance_id',
                                                                                    ondelete='CASCADE',
                                                                                    onupdate='cascade'),
                                                                            nullable=False)
    national_facility_and_generic_sector_id = Column(Integer(),
                                                    ForeignKey('facility.national_facility_and_generic_sector_id',
                                                            ondelete='CASCADE',
                                                            onupdate='cascade'),
                                                    nullable=False)
    national_generic_transfer_class_id = Column(Integer(),
                                                ForeignKey('national_generic_transfer_class.national_generic_transfer_class_id',
                                                        ondelete='CASCADE',
                                                        onupdate='cascade'),
                                                nullable=False)
    transfer_amount_kg = Column(Float(precision=2), nullable=False)
    reliability_score = Column(Integer(), nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    national_generic_substance = relationship("NationalGenericSubstance",
                                            back_populates="transfer_record")
    facility = relationship("Facility",
                            back_populates="transfer_record")
    national_generic_transfer_class = relationship("NationalGenericTransferClass",
                                            back_populates="transfer_record")

    def __init__(self, **kwargs):
        self.transfer_record_id = kwargs['transfer_record_id']
        self.reporting_year = kwargs['reporting_year']
        self.national_generic_substance_id = kwargs['national_generic_substance_id']
        self.national_facility_and_generic_sector_id = kwargs['national_facility_and_generic_sector_id']
        self.national_generic_transfer_class_id = kwargs['national_generic_transfer_class_id']
        self.transfer_amount_kg = kwargs['transfer_amount_kg']
        self.reliability_score = kwargs['reliability_score']