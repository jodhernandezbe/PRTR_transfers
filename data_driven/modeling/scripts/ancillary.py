
# Importing libraries
from data_driven.modeling.scripts.evaluation import performing_cross_validation, y_randomization

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import re
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence


def drive_authentication():
    '''
    Function to authenticate with Google Drive

    Output:
     - drive: GoogleDrive instance
    '''

    credentials_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                os.pardir, 'notebooks',
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


def cv_and_y_random(id, X_train, Y_train, model_params, model, classification_type, threshold=0.75, stopping_metric='val_f1'):
    '''
    Function to test the RFC with defined params
    
    Input:
        - id: string = data processing id
        - X_train, Y_train: numpy arrays
        
    Output:
        - df_result: dataframe = model evaluation under Y-randomization and cross-validation
    '''
                                                       
    # 5-fold cross-validation
    cv_result = performing_cross_validation(model,
                                           model_params,
                                           X_train,
                                           Y_train,
                                           classification_type,
                                           threshold=threshold,
                                           stopping_metric=stopping_metric)
    df_cv = pd.DataFrame({key: [val] for key, val in cv_result.items()})
                               
    # Y-randomization
    y_randomization_error = y_randomization(model,
                                       model_params,
                                       X_train,
                                       Y_train,
                                       classification_type,
                                       threshold=threshold,
                                       stopping_metric=stopping_metric)
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

    path_pickle = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        os.pardir,
                        'output', 'models',
                        classification_type.replace(' ', '_'),
                        'RFC_best_params.pkl')

    if os.path.isfile(path_pickle):
        mode = 'ab'
    else:
        mode = 'wb'

    with open(path_pickle, mode) as outfile:
        pickle.dump(best_params, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def rfc_draw_convergence(tuning_result, classification_type):
    '''
    Function to plot the convergence of the hyperparameters tuning for RFC

    Input:
        - tuning_result: dictionary
        - path_plot: string

    Output:
        - None
    '''

    fig, ax = plt.subplots(figsize=(14,5))

    # Convergence plot
    plot_convergence(tuning_result['search'], ax=ax)

    # Title
    ax.set_title(f'Convergence of Bayesian Optimization for RFC model and {classification_type}\n',
                fontsize=18, fontweight='bold')

    # Organize horizontal grid
    ax.grid(axis='y') 
    ax.set_axisbelow(True)

    # Remove top and right boders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # axis labels
    ax.set_ylabel('- Mean F1 score', fontsize=16, fontweight='bold', labelpad=20)
    ax.set_xlabel('# Interations', fontsize=16, fontweight='bold', labelpad=20)

    # Y ticks fontsize
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    # X ticks fontsize
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    # Set the x-axis limit
    plt.xlim(1,len(tuning_result['search'].func_vals))
    
    # Save the plot
    path_plot = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        os.pardir,
                        'output', 'figures',
                        classification_type.replace(' ', '_'),
                        'RFC_convergence.pdf')
    plt.savefig(path_plot, dpi=fig.dpi, bbox_inches='tight')
    
