#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from pydantic import BaseModel


class RequestModel(BaseModel):
    smiles: str
    transfer_amount_kg: float
    generic_sector_code: int
    epsi: float
    gva: float
    price_usd_g: float