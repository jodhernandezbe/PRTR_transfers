from fastapi import FastAPI, Query

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


'''
Multi-label classification
'''
@app.post('/v1/mlc_classification/',
        summary='Multi-label classification predictions',
        tags=['Multi-label classification'])
async def mlc_classification(
                prob: bool = Query(False)
                          ):     
    return {
        'mlc_classification': 'mlc_classification'
    }

'''
Multi-model binary classification
'''