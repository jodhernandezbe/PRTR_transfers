#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from pydantic import BaseModel
from typing import Dict, Optional


class RequestModel(BaseModel):
    smiles: str
    transfer_amount_kg: float
    generic_sector_code: int
    epsi: float
    gva: float
    price_usd_g: float


class ResponseModel(BaseModel):
    belong_to: Dict[str, bool]
    probability: Optional[Dict[str, float]] = None