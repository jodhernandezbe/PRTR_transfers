#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_engineering.load.base import Base

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship


class NationalGenericTransferClass(Base):
    __tablename__ = 'national_generic_transfer_class'

    national_generic_transfer_class_id = Column(Integer(), primary_key=True)
    generic_transfer_class_id = Column(String(3), 
                                       ForeignKey('generic_transfer_class.generic_transfer_class_id',
                                                ondelete='CASCADE',
                                                onupdate='cascade'),
                                       nullable=False)
    national_transfer_class_prtr_system_id = Column(Integer(), 
                                                ForeignKey('national_transfer_class.national_transfer_class_prtr_system_id',
                                                        ondelete='CASCADE',
                                                        onupdate='cascade'),
                                                nullable=False)
    created_at = Column(DateTime(), default=datetime.now())

    national_transfer_class = relationship("NationalTransferClass",
                                            back_populates="national_generic_transfer_class")
    generic_transfer_class = relationship("GenericTransferClass",
                                        back_populates="national_generic_transfer_class")
    transfer_record = relationship("TransferRecord",
                                    back_populates="national_generic_transfer_class")

    def __init__(self, **kwargs):
        self.national_generic_transfer_class_id = kwargs['national_generic_transfer_class_id']
        self.generic_transfer_class_id = kwargs['generic_transfer_class_id']
        self.national_transfer_class_prtr_system_id = kwargs['national_transfer_class_prtr_system_id']


class NationalTransferClass(Base):
    __tablename__ = 'national_transfer_class'

    national_transfer_class_prtr_system_id = Column(Integer(), primary_key=True)
    national_transfer_class_id = Column(String(6), nullable=False)
    national_transfer_class_name = Column(String(150), nullable=False)
    prtr_system = Column(String(5), nullable=False)
    prtr_system_comment = Column(String(200), nullable=True)
    created_at = Column(DateTime(), default=datetime.now())

    national_generic_transfer_class = relationship("NationalGenericTransferClass",
                                            back_populates="national_transfer_class",
                                            uselist=False)

    def __init__(self, **kwargs):
        self.national_transfer_class_prtr_system_id = kwargs['national_transfer_class_prtr_system_id']
        self.national_transfer_class_id = kwargs['national_transfer_class_id']
        self.national_transfer_class_name = kwargs['national_transfer_class_name']
        self.prtr_system = kwargs['prtr_system']
        self.prtr_system_comment = kwargs['prtr_system_comment']


class GenericTransferClass(Base):
    __tablename__ = 'generic_transfer_class'

    generic_transfer_class_id = Column(String(3), primary_key=True)
    generic_transfer_class_name = Column(String(250), nullable=False)
    transfer_class_wm_hierarchy_name = Column(String(20), nullable=False)
    generic_system_comment = Column(String(150), nullable=True)
    created_at = Column(DateTime(), default=datetime.now())

    national_generic_transfer_class = relationship("NationalGenericTransferClass",
                                            back_populates="generic_transfer_class")

    def __init__(self, **kwargs):
        self.generic_transfer_class_id = kwargs['generic_transfer_class_id']
        self.generic_transfer_class_name = kwargs['generic_transfer_class_name']
        self.transfer_class_wm_hierarchy_name = kwargs['transfer_class_wm_hierarchy_name']
        self.generic_system_comment = kwargs['generic_system_comment']