#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, HTTPException
from fastapi.encoders import jsonable_encoder
from typing import Optional, List

from model import RequestModel
from estimations import get_estimations

app = FastAPI(title='PRTR Transfers Model Deployment',
            version='1',
            contact={
                    "name": "Jose D. Hernandez-Betancur",
                    "url": "www.sustineritechem.com",
                    "email": "jodhernandezbemj@gmail.com"
                    },
            license_info={
                    "name": "GNU General Public License v3.0",
                    "url": "https://github.com/jodhernandezbe/PRTR_transfers/blob/model-deployment/LICENSE",
                        },
            docs_url="/v1/api_documentation",
            redoc_url=None,
            description=f'''
            This is an API service containing the model deployment for the PRTR transfers. The models provided are Random Forest Classifiers, using both a multi-label classification strategy and a multi-model binary classification strategy (or one-vs-all). The target variable values are the 10 transfer classes presented in the following <a href="https://prtr-transfers-summary.herokuapp.com/transfer_classes/" rel="noopener noreferrer" target="_blank">link</a>.
            ''')


###############################################################################################
# Multi-label classification
###############################################################################################

@app.post('/v1/mlc_classification/',
        summary='Multi-label classification predictions',
        tags=['Multi-label classification'])
async def mlc_classification(
                        input_features: RequestModel,
                        prob: Optional[bool] = Query(False)
                          ):     

        input_features_dict = input_features.dict()

        estimates = get_estimations(input_features_dict,
                        prob=prob)

        if not estimates:
                raise HTTPException(status_code=404,
                                detail="Descriptors for the input SMILES not found in RDKit.")

        json_compatible_estimates = jsonable_encoder(estimates)

        return JSONResponse(content=json_compatible_estimates)

###############################################################################################
# Multi-model binary classification
###############################################################################################

@app.post('/v1/mmbc_classification/',
        summary='Multi-modlel binary classification (one-vs-all) predictions',
        tags=['Multi-modlel binary classification (one-vs-all)'])
async def mlc_classification(
                        input_features: RequestModel,
                        prob: Optional[bool] = Query(False),
                        transfer_class: Optional[List[str]] = Query(['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10'])
                          ):

        input_features_dict = input_features.dict()


        estimates = get_estimations(input_features_dict,
                        prob=prob,
                        transfer_class=transfer_class
                        )

        if not estimates:
                raise HTTPException(status_code=404,
                                detail="Descriptors for the input SMILES not found in RDKit.")

        json_compatible_estimates = jsonable_encoder(estimates)

        return JSONResponse(content=json_compatible_estimates)