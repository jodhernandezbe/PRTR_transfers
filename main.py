#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional, List

from model import RequestModel, ResponseModel
from estimations import get_estimations

app = FastAPI(title='PRTR Transfers Model Deployment',
            version='1',
            contact={
                    "name": "Jose D. Hernandez-Betancur",
                    "url": "www.sustainableprosys.com",
                    "email": "jodhernandezbemj@gmail.com"
                    },
            license_info={
                    "name": "GNU General Public License v3.0",
                    "url": "https://github.com/jodhernandezbe/PRTR_transfers/blob/model-deployment/LICENSE",
                        },
            docs_url="/v1/api_documentation",
            redoc_url=None,
            description=f'''This is an API service providing the models deployed for the PRTR Transfers project. The models are Random Forest Classifiers, and the modeling strategy uses both multi-label classification and a multi-model binary classification strategy (or one-vs-all). TThe target variable values are the 10 transfer classes presented in the following <a href="https://prtr-transfers-summary.herokuapp.com/transfer_classes/" rel="noopener noreferrer" target="_blank">link</a>. The applicability domain of the model involves the industry sectors presented in the following <a href="https://prtr-transfers-summary.herokuapp.com/sectors/" rel="noopener noreferrer" target="_blank">link</a>. Read the below documentation for more information about the services offered.
            ''')


###############################################################################################
# Multi-label classification
###############################################################################################

@app.post('/v1/mlc_classification/',
        response_model=ResponseModel,
        summary='Multi-label classification predictions',
        tags=['Multi-label classification'],
        responses={
                    404: {'description': 'Descriptors for the input SMILES not found in RDKit.'},
                    200: {'description': 'JSON output obtained based on user input parameter(s).',
                        'content': {
                                    'application/json': {
                                        'example': {
                                                "belong_to": {
                                                        "M1": False,
                                                        "M2": False,
                                                        "M3": False,
                                                        "M4": False,
                                                        "M5": False,
                                                        "M6": False,
                                                        "M7": False,
                                                        "M8": False,
                                                        "M9": False,
                                                        "M10": False
                                                },
                                                "probability": {
                                                        "M1": 0.07,
                                                        "M2": 0.16,
                                                        "M3": 0.13,
                                                        "M4": 0.24,
                                                        "M5": 0.13,
                                                        "M6": 0.17,
                                                        "M7": 0.2,
                                                        "M8": 0.44,
                                                        "M9": 0.12,
                                                        "M10": 0.35
                                                }
                                                        }
                                                        }
                                    }
                        }
                    })
async def mlc_classification(
                        input_features: RequestModel = Body(..., example={'smiles': 'C1=CC=CC=C1',
                                                                'transfer_amount_kg': 10.0,
                                                                'generic_sector_code': 10,
                                                                'epsi': 0.53,
                                                                'gva': 100.0,
                                                                'price_usd_g': 3.0}),
                        prob: Optional[bool] = Query(False)
                          ):
        '''
        Multi-label classification predictions

        This function is used to obtain the predictions for the input features using the Random Forest Classifier.

        Parameters:
        - Request body parameters:
                <ul>
                        <li>smiles (str): SMILES string of the molecule</li>
                        <li>transfer_amount_kg (float): Transfer amount in kg/yr</li>
                        <li>generic_sector_code (int): Generic sector code</li>
                        <li>epsi (float): OECD Environmental Policy Stringency index for your country</li>
                        <li>gva (float): Gross value added for your sector in your country in USD/yr</li>
                        <li>price_usd_g (float): Price of the chemical in USD/g</li>
                </ul>
        - Query parameters:
                <ul>
                        <li>prob (bool): If True, the probability of belonging to each class is returned. Default is False</li>
                </ul>

        Returns a JSON response with the following fields:
        - belong_to (dict): Dictionary with the 10 transfer classes as keys and a boolean value indicating if the chemical belongs to that class as value.
        - probability (dict): Dictionary with the 10 transfer classes as keys and the probability of belonging to that class as value.
        '''  

        input_features_dict = input_features.dict()
        input_features_dict['smiles'] = input_features_dict['smiles'].upper()

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
        response_model=ResponseModel,
        summary='Multi-modlel binary classification (one-vs-all) predictions',
        tags=['Multi-model binary classification (one-vs-all)'],
        responses={
                    404: {'description': 'Descriptors for the input SMILES not found in RDKit.'},
                    200: {'description': 'JSON output obtained based on user input parameter(s).',
                        'content': {
                                    'application/json': {
                                        'example': {
                                                "belong_to": {
                                                        "M1": False,
                                                        "M2": False,
                                                        "M3": False,
                                                        "M4": False,
                                                        "M5": False,
                                                        "M6": False,
                                                        "M7": False,
                                                        "M8": False,
                                                        "M9": False,
                                                        "M10": False
                                                },
                                                "probability": {
                                                        "M1": 0.25,
                                                        "M2": 0.52,
                                                        "M3": 0.49,
                                                        "M4": 0.52,
                                                        "M5": 0.36,
                                                        "M6": 0.33,
                                                        "M7": 0.28,
                                                        "M8": 0.47,
                                                        "M9": 0.29,
                                                        "M10": 0.6
                                                }
                                                        }
                                                        }
                                    }
                        }
                    })
async def mlc_classification(
                        input_features: RequestModel = Body(..., example={'smiles': 'C1=CC=CC=C1',
                                                                'transfer_amount_kg': 10.0,
                                                                'generic_sector_code': 10,
                                                                'epsi': 0.53,
                                                                'gva': 100.0,
                                                                'price_usd_g': 3.0}),
                        prob: Optional[bool] = Query(False),
                        transfer_class: Optional[List[str]] = Query(['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10'])
                          ):
        '''
        Multi-model binary classification (one-vs-all) predictions

        This function is used to obtain the predictions for the input features using the Random Forest Classifier.

        Parameters:
        - Request body parameters:
                <ul>
                        <li>smiles (str): SMILES string of the molecule</li>
                        <li>transfer_amount_kg (float): Transfer amount in kg/yr</li>
                        <li>generic_sector_code (int): Generic sector code</li>
                        <li>epsi (float): OECD Environmental Policy Stringency index for your country</li>
                        <li>gva (float): Gross value added for your sector in your country in USD/yr</li>
                        <li>price_usd_g (float): Price of the chemical in USD/g</li>
                </ul>
        - Query parameters:
                <ul>
                        <li>prob (bool): If True, the probability of belonging to each class is returned. Default is False</li>
                        <li>transfer_class (list): List of the 10 transfer classes to be used.</li>
                </ul>

        Returns a JSON response with the following fields:
        - belong_to (dict): Dictionary with the desired transfer classes as keys and a boolean value indicating if the chemical belongs to that class as value.
        - probability (dict): Dictionary with the desired transfer classes as keys and the probability of belonging to that class as value.
        '''

        input_features_dict = input_features.dict()
        input_features_dict['smiles'] = input_features_dict['smiles'].upper()


        estimates = get_estimations(input_features_dict,
                        prob=prob,
                        transfer_class=transfer_class
                        )

        if not estimates:
                raise HTTPException(status_code=404,
                                detail="Descriptors for the input SMILES not found in RDKit.")

        json_compatible_estimates = jsonable_encoder(estimates)

        return JSONResponse(content=json_compatible_estimates)