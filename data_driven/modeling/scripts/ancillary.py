
# Importing libraries
from data_driven.modeling.scripts.evaluation import performing_cross_validation, y_randomization

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import re
import numpy as np
import pandas as pd
import json 

def drive_authentication():
    '''
    Function to authenticate with Google Drive

    Output:
     - drive: GoogleDrive instance
    '''

    credentials_file = os.path.join(os.getcwd(),
                                os.pardir,
                               'client_secrets.json')
    GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = credentials_file
    GoogleAuth.DEFAULT_SETTINGS['oauth_scope'] = ['https://www.googleapis.com/auth/drive']


    # Authenticating
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
    drive = GoogleDrive(gauth)

    return drive


def npz_files(parent_id, drive):
    '''
    Function to list the files in a Google Drive folder
    
    Input:
        - parent_id: string
        - drive: GoogleDrive instance
    
    Output:
        - ids: Python dictionary whose keys are the pipeline ids and values the id of the file in Google Drive
    '''
    
    filelist=[]
    file_list = drive.ListFile({'q': "'%s' in parents and trashed=false" % parent_id}).GetList()
    
    for f in file_list:
        if f['mimeType']=='application/vnd.google-apps.folder': # if folder
            filelist.append({"id":f['id'],"title":f['title'],"list":npz_files(f['id'])})
        else:
            filelist.append({"title":f['title'],"title1":f['alternateLink']})

    ids = {re.search(r'([1-9]{1,2}).npz', file['title']).group(1):
       re.search(r'https://.*/d/(.*)/view?.*', file['title1']).group(1)
       for file in filelist
       if file['title'].endswith('.npz')}
            
    return ids


def open_dataset(id, key, drive, test=False):
    '''
    Function open the data sets:
    
    Input:
        - id: string = data processing id
        - key: string = key or id for the file in Google Drive
        - test: boolean = if the test set is required
        - drive: GoogleDrive instance
        
    Output:
        - X_train, Y_train, X_test, Y_test: numpy arrays
    '''
    
    # Getting the from Google Drive
    f_data = drive.CreateFile({'id': key})
    f_data.GetContentFile(f'{id}.npz')
                             
    # Loading the .npz for training
    with np.load(f'{id}.npz') as data:
        
        X_train = data['X_train']
        Y_train = data['Y_train']
        # Checking the dimensions
        print(f'X train has the following dimensions: {X_train.shape}')
        print(f'Y train has the following dimensions: {Y_train.shape}')
        
        if test:
            
            X_test = data['X_test']
            Y_test = data['Y_test']
            
            # Checking the dimensions
            print(f'X test has the following dimensions: {X_test.shape}')
            print(f'Y test has the following dimensions: {Y_test.shape}')
            
            os.remove(f'{id}.npz')
            
            return X_train, Y_train, X_test, Y_test
        
        os.remove(f'{id}.npz')
        
        return X_train, Y_train


def build_base_model(id, X_train, Y_train, model_params, model, classification_type):
    '''
    Function to test the RFC with default params
    
    Input:
        - id: string = data processing id
        - X_train, Y_train: numpy arrays
        
    Output:
        - df_result: dataframe = model evaluation under Y-randomization and cross-validation
    '''
                               
    # Default parameters
    model_params = {
        'bootstrap': True,
        'ccp_alpha': 0.0,
        'class_weight': 'balanced',
        'criterion': 'gini',
        'max_depth': None,
        'max_features': 'sqrt',
        'max_leaf_nodes': None,
        'max_samples': None,
        'min_impurity_decrease': 0.0,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'min_weight_fraction_leaf': 0.0,
        'n_estimators': 100,
        'n_jobs': 4,
        'oob_score': False,
        'random_state': 0,
        'verbose': 0,
        'warm_start': False,
    }
                               
    # 5-fold cross-validation
    cv_result = performing_cross_validation(model,
                                           model_params,
                                           X_train,
                                           Y_train.reshape(Y_train.shape[0],),
                                           classification_type)
    df_cv = pd.DataFrame({key: [val] for key, val in cv_result.items()})
                               
    # Y-randomization
    y_randomization_error = y_randomization('RFC',
                                       model_params,
                                       X_train,
                                       Y_train.reshape(Y_train.shape[0],),
                                       'multi-class classification')
    df_yr = pd.DataFrame({key: [val] for key, val in y_randomization_error.items()})
                               
    
    df_result = pd.concat([df_cv, df_yr], axis=1)
    df_result['id'] = id
    
    return df_result


def save_best_params(classification_type, best_params):
    '''
    Function to save best parameter after tuning

    Input:
        - classification_type: string
        - best_params: dictionary
    '''

    path_json = os.path.join(os.getcwd(),
                        os.pardir, os.pardir,
                        'output', 'models', classification_type,
                        'best_params.json')

    if os.path.isfile(path_json):
        mode = 'a'
    else:
        mode = 'w'

    with open(path_json, mode) as outfile:
        json.dump(best_params, outfile, cls=NpEncoder)
    

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)