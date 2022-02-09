#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from data_driven.modeling.scripts.models import defining_model, StoppingDesiredANN
from data_driven.modeling.scripts.metrics import prediction_evaluation

from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.model_selection import IterativeStratification as KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy import stats
from tqdm import tqdm
import time
import os


def performing_cross_validation(model, model_params, X, Y, classification_type, for_tuning=False, threshold_ann=0.75, stopping_metric='val_f1'):
    '''
    Function to apply k-fold cross validation
    '''
    if not for_tuning:
        pbar = tqdm(desc='5-fold cross validation', total=5, initial=0)

    if classification_type == 'multi-label classification':
        loss_metric = '0_1_loss'
        kf = KFold(n_splits=5, random_state=None, order=1)
    else:
        loss_metric = 'error'
        kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)

    validation_acc = []
    validation_0_1_loss_or_error = []
    validation_f1 = []
    train_acc = []
    train_f1 = []

    for train_index , validation_index in kf.split(X, Y):
        X_train , X_validation = X[train_index], X[validation_index]
        Y_train , Y_validation = Y[train_index] , Y[validation_index]

        # Instantiating the model
        classifier = defining_model(model, model_params)

        # Fitting the model
        if model == 'RFC':
            classifier.fit(X_train, Y_train)
            Y_train_hat = classifier.predict(X_train)
            Y_validation_hat = classifier.predict(X_validation)
        else:
            classifier.fit(X_train, Y_train,
                      batch_size=model_params['batch_size'],
                      verbose=model_params['verbose'],
                      epochs=model_params['epochs'],
                      shuffle=model_params['shuffle'],
                      callbacks=[StoppingDesiredANN(threshold=threshold_ann,
                                                metric=stopping_metric),
                        EarlyStopping(monitor='val_loss',
                                        min_delta=1e-4,
                                        patience=10,
                                        verbose=0,
                                        mode='auto'),
                        ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.1,                 
                                        patience=5,
                                        verbose=0,
                                        mode='auto',
                                        min_delta=1e-4)])

            # Predicting the validation set and evaluating the model
            if classification_type == 'multi-class classification':
                Y_train_hat = np.argmax(classifier.predict(X_train), axis=1)
                Y_validation_hat = np.argmax(classifier.predict(X_validation), axis=1)
            else:
                Y_train_hat = np.where(classifier.predict(X_train) > 0.5, 1, 0)
                Y_validation_hat = np.where(classifier.predict(X_validation) > 0.5, 1, 0)

        # Validation set evaluation
        validation_f1.append(
            prediction_evaluation(Y_validation, Y_validation_hat, metric='f1')
        )

        if for_tuning:
            pass
        else:
            validation_acc.append(
                prediction_evaluation(Y_validation, Y_validation_hat)
            )
            validation_0_1_loss_or_error.append(
                prediction_evaluation(Y_validation, Y_validation_hat, metric=loss_metric)
            )

            # Train set evaluation
            train_f1.append(
                prediction_evaluation(Y_train, Y_train_hat, metric='f1')
            )
            train_acc.append(
                prediction_evaluation(Y_train, Y_train_hat)
            )
    
        del Y_validation_hat, Y_train_hat

        if not for_tuning:
            time.sleep(0.1)
            pbar.update(1)

    if not for_tuning:
        pbar.close()

    # Summaries
    if not for_tuning:

        mean_validation_acc = round(np.mean(np.array(validation_acc)), 2)
        mean_train_acc = round(np.mean(np.array(train_acc)), 2)
        accuracy_analysis = overfitting_underfitting(
                                            np.array(train_acc),
                                            np.array(validation_acc)
                                            )
        mean_validation_f1 = round(np.mean(np.array(validation_f1)), 2)
        mean_train_f1 = round(np.mean(np.array(train_f1)), 2)
        mean_validation_0_1_loss_or_error = round(np.mean(np.array(validation_0_1_loss_or_error)), 2)
        std_validation_0_1_loss_or_error = round(np.std(np.array(validation_0_1_loss_or_error)), 6)

        cv_result = {'mean_validation_accuracy': mean_validation_acc,
                'mean_train_accuracy': mean_train_acc,
                'accuracy_analysis': accuracy_analysis,
                'mean_validation_f1': mean_validation_f1,
                'mean_train_f1': mean_train_f1,
                'mean_validation_0_1_loss_or_error': mean_validation_0_1_loss_or_error,
                'std_validation_0_1_loss_or_error': std_validation_0_1_loss_or_error}
    else:

        cv_result = round(np.mean(np.array(validation_f1)), 2)

    return cv_result


def overfitting_underfitting(score_train, score_test):
    '''
    Funtion to determine overfitting and underfitting

    Conditions:
    
    1. High training accurracy (>= 0.75)
    2. Small gap between accucaries

    The non-parametric hypothesis test: Mann Whitney U Test (Wilcoxon Rank Sum Test)

    H0: The score_train and  score_test are equal
    H1: The score_train and  score_test are not equal

    5% level of significance (i.e., α=0.05)
    '''

    mean_score_train = round(np.mean(score_train), 2)
    U1, p = mannwhitneyu(score_train,
                        score_test,
                        method="exact")

    if mean_score_train < 0.75:
        if p < 0.05:
            return 'under-fitting (high bias and high variance)'
        else:
            return 'under-fitting (high bias)'
    else:
        if p < 0.05:
            return 'over-fitting (high variance)'
        else:
            return 'optimal-fitting'


def y_randomization(model, model_params, X, Y, classification_type, threshold_ann=0.75, stopping_metric='val_f1'):
    '''
    Function to apply Y-Randomization
    '''


    if classification_type == 'multi-label classification':
        loss_metric = '0_1_loss'
    else:
        loss_metric = 'error'

    shuffled_losses = []

    # Splitting the data
    if classification_type == 'multi-label classification':
        Y = np.array([[int(element) for element in row.split(' ')] for row in Y])
        X_train, Y_train, X_validation, Y_validation = iterative_train_test_split(X,
                                                                                Y,
                                                                                test_size=0.2)
    else:
        X_train, X_validation, Y_train, Y_validation = train_test_split(X,
                                                Y,
                                                test_size=0.2,
                                                random_state=0,
                                                shuffle=True,
                                                stratify=Y)

    indexes_train = list(range(Y_train.shape[0]))
    indexes_validation = list(range(Y_validation.shape[0]))

    for i in tqdm(range(0, 10), initial=0,  desc="Y-Randomization"):

        # Shuffling the labels
        np.random.shuffle(indexes_train)
        np.random.shuffle(indexes_validation)

        classifier = defining_model(model, model_params)

        # Fitting the model
        if model == 'RFC':
            classifier.fit(X_train, Y_train[indexes_train])
            Ypred = classifier.predict(X_validation)
        else:
            classifier.fit(X_train, Y_train[indexes_train],
                      batch_size=model_params['batch_size'],
                      verbose=model_params['verbose'],
                      epochs=model_params['epochs'],
                      shuffle=model_params['shuffle'],
                      callbacks=[StoppingDesiredANN(threshold=threshold_ann,
                                                metric=stopping_metric),
                        EarlyStopping(monitor='val_loss',
                                        min_delta=1e-4,
                                        patience=10,
                                        verbose=0,
                                        mode='auto'),
                        ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.1,                 
                                        patience=5,
                                        verbose=0,
                                        mode='auto',
                                        min_delta=1e-4)])
            if classification_type == 'multi-class classification':
                Ypred = np.argmax(classifier.predict(X_validation), axis=1)
            else:
                Ypred = np.where(classifier.predict(X_validation) > 0.5, 1, 0)

        shuffled_losses.append(prediction_evaluation(Y_validation[indexes_validation],
                                                    Ypred,
                                                    metric=loss_metric))
    
    y_randomization_error = {'y_randomization_mean_0_1_loss_or_error': round(np.mean(np.array(shuffled_losses)), 2),
                            'y_randomization_std_0_1_loss_or_error': round(np.std(np.array(shuffled_losses)), 6)}

    return y_randomization_error


def data_driven_models_ranking(df):
    '''
    Function to select the data driven model

    Here a best-worst scaling is applied as well
    criterion = (score -  score_worst)/(score - score_best)
    '''

    dict_for_acc = {'optimal-fitting': 4,
                    'over-fitting (high variance)': 3,
                    'under-fitting (high bias)': 2,
                    'under-fitting (high bias and high variance)': 1
                    }

    # Criterion 1
    df['criterion_1'] = df[18].apply(lambda x: dict_for_acc[x])
    df['criterion_1'] = (df['criterion_1'] - 1)/(4 - 1)

    # Criterion 2
    df['criterion_2'] = df[16]
    df['criterion_2'] = (df['criterion_2'] - 0)/(1 - 0)

    # Criterion 3
    df.loc[df[26] == 0, 26] = 10 ** -4
    df['criterion_3'] = df[25]/df[26]
    df['criterion_3'] = (df['criterion_3'] - df['criterion_3'].max())/(df['criterion_3'].min() - df['criterion_3'].max())
    
    # Criterion 4
    df['criterion_4'] = df[19]/(df[23] - df[24])
    df['criterion_4'] = (df['criterion_4'] - df['criterion_4'].max())/(0 - df['criterion_4'].max())

    # FAHP
    df =  df[[f'criterion_{i}' for i in range(1,5)]]
    df = fahp(df)

    df['rank'] = df['weight'].rank(method='dense', ascending=False).astype(int)

    return df['rank']


def comparison_matrix(df):
    '''
    Function to generate the comparison matrix
    '''

    m_criteria = 0
    n = df.shape[0]
    for col in df.columns:
        m_criteria = m_criteria + 1
        val = df[col].tolist()
        N_aux = np.empty((n, n))
        for i in range(n):
            score_i = val[i]
            for j in range(i, n):
                score_j = val[j]
                diff = score_i - score_j
                diff = round(4*(diff - 1) + 4)
                N_aux[i][j] = diff
        if m_criteria == 1:
            N = N_aux
        else:
            N = np.concatenate((N, N_aux), axis=0)

    return N


def fahp(df):
    '''
    Function to apply Fuzzy Analytic Hierarchy Process (FAHP)

    Triangular fuzzy numbers (TFN)
    n is number of combinations
    TFN is a vector with n segments and each one has 3 different numbers
    The segments are the linguistic scales
    The 3 differente numbers in the segments are a triangular fuzzy number (l,m,u)
    the first segment: equal importance; the second one: moderate importance of one over another;
    the third one: strong importance of one over another; the fourth one: very strong importance of one over another
    the fifth one: Absolute importance of one over another
    '''

    n = df.shape[0]
    m_criteria = df.shape[1]
    N = comparison_matrix(df)
    # Definition of variables
    W = np.zeros((1, n))  # initial value of weights vector (desired output)
    w = np.zeros((m_criteria, n))  # initial value of weithts matrix
    phi = np.zeros((m_criteria, 1))  # diversification degree
    eps = np.zeros((m_criteria, 1))  # entropy
    theta = np.zeros((1, m_criteria))  # criteria' uncertainty degrees
    sumphi = 0  # Initial value of diversification degree

    TFN = np.array([1, 1, 1, 2/3, 1, 3/2, 3/2, 2,
                    5/2, 5/2, 3, 7/2, 7/2, 4, 9/2])
    for k in range(1, m_criteria + 1):
        a = np.zeros((1, n*n*3))  # Comparison matrix (In this case is a vector because of computational memory
        for i in range(k*n-(n-1), k*n + 1):
            for j in range(i-n*(k-1), n + 1):
                # This is the position of the third element of the segment for
                # a*(i,j) (upper triangular part)
                jj = 3*(n*((i-n*(k-1))-1)+j)
                # This is the position of the thrid element of the segment for
                # a*(j,i) (lower triangular part)
                jjj = 3*(n*(j-1) + i-n*(k-1))
                if N[i - 1][j - 1] == -4:
                    a[0][jjj-3:jjj] = TFN[12:15]
                    a[0][jj-3:jj] = np.array([TFN[14]**-1, TFN[13]**-1, TFN[12]**-1])
                elif N[i - 1][j - 1] == -3:
                    a[0][jjj-3:jjj] = TFN[9:12]
                    a[0][jj-3:jj] = np.array([TFN[11]**-1, TFN[10]**-1, TFN[9]**-1])
                elif N[i - 1][j - 1] == -2:
                    a[0][jjj-3:jjj] = TFN[6:9]
                    a[0][jj-3:jj] = np.array([TFN[8]**-1, TFN[7]**-1, TFN[6]**-1])
                elif N[i - 1][j - 1] == -1:
                    a[0][jjj-3:jjj] = TFN[3:6]
                    a[0][jj-3:jj] = np.array([TFN[5]**-1, TFN[4]**-1, TFN[3]**-1])
                elif N[i - 1][j - 1] == 0:
                    a[0][jj-3:jj] = TFN[0:3]
                    a[0][jjj-3:jjj] = TFN[0:3]
                elif N[i - 1][j - 1] == 1:
                    a[0][jj-3:jj] = TFN[3:6]
                    a[0][jjj-3:jjj] = np.array([TFN[5]**-1, TFN[4]**-1, TFN[3]**-1])
                elif N[i - 1][j - 1] == 2:
                    a[0][jj-3:jj] = TFN[6:9]
                    a[0][jjj-3:jjj] = np.array([TFN[8]**-1, TFN[7]**-1, TFN[6]**-1])
                elif N[i - 1][j - 1] == 3:
                    a[0][jj-3:jj] = TFN[9:12]
                    a[0][jjj-3:jjj] = np.array([TFN[11]**-1, TFN[10]**-1, TFN[9]**-1])
                elif N[i - 1][j - 1] == 4:
                    a[0][jj-3:jj] = TFN[12:15]
                    a[0][jjj-3:jjj] = np.array([TFN[14]**-1, TFN[13]**-1, TFN[12]**-1])
        # (2) fuzzy synthetic extension
        A = np.zeros((n, 3))
        B = np.zeros((1, 3))
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                jj = 3*(n*(i-1)+j)
                A[i - 1][:] = A[i - 1][:] + a[0][jj-3:jj]
            B = B + A[i - 1][:]
        BB = np.array([B[0][2]**-1, B[0][1]**-1, B[0][0]**-1])
        S = A*BB
        # (3) Degree of possibility
        for i in range(n):
            V = np.zeros(n)
            for j in range(n):
                if S[i][1] >= S[j][1]:
                    V[j] = 1
                elif S[j][0] >= S[i][2]:
                    V[j] = 0
                else:
                    V[j] = (S[j][0] - S[i][2])/((S[i][1] - S[i][2]) - (S[j][1] - S[j][0]))
            w[k - 1][i] = np.min(V)
        # (4) Weight of each flow for a criterium
        w[k - 1][:] = (np.sum(w[k - 1][:])**-1)*w[k - 1][:]
        # (5) Criteria' uncertainty degrees
        for i in range(n):
            if w[k - 1][i] != 0:
                eps[k - 1][0] = eps[k - 1][0] - w[k - 1][i]*np.log(w[k - 1][i])
        eps[k - 1][0] = eps[k - 1][0]/np.log(n)
        phi[k - 1][0] = 1 + eps[k - 1]
        sumphi = sumphi + phi[k - 1][0]
    # (6) Final weight of all flows
    for i in range(n):
        for k in range(m_criteria):
            theta[0][k] = phi[k][0]/sumphi
            W[0][i] = W[0][i] + w[k][i]*theta[0][k]
    # Final weights
    W = (np.sum(W)**-1)*W
    W = W.T
    df = df.assign(weight=W)
    return df


def calc_distance(dimensionality_reduction_method,
                dimensionality_reduction,
                X, centroid):
    '''
    Function to calculate the distance between data centroid and any point
    '''

    if (not dimensionality_reduction) or (dimensionality_reduction_method != 'FAMD'):
        # Ahmad & dey’s distance
        numerical_vals = centroid.loc[centroid['central_tendency'] == 'mean', 'centroid'].index.tolist()
        distance_numerical = np.sum((X[:, numerical_vals] - centroid.loc[numerical_vals, 'centroid'].values) ** 2, axis=1) ** 0.5
        categorical_vals = centroid.loc[centroid['central_tendency'] == 'mode', 'centroid'].index.tolist()
        func = lambda x: 1 if not x else 0
        func = np.vectorize(func)
        matrix_caterorical = X[:, categorical_vals].round(2) == centroid.loc[categorical_vals, 'centroid'].values.round(2)
        matrix_caterorical = func(matrix_caterorical)
        distance_categorical = np.sum(matrix_caterorical, axis=1)
        distances = distance_numerical + distance_categorical
    else:
        # Euclidean distance
        distances = np.sum((X - centroid['centroid'].values) ** 2, axis=1) ** 0.5
        
    return distances


def centroid_cal(dimensionality_reduction_method,
                dimensionality_reduction,
                X, feature_cols, num_cols):
    '''
    Function to calculate data centroid
    '''

    if (not dimensionality_reduction) or (dimensionality_reduction_method != 'FAMD'):
        col_central = list(np.mean(X[:, [i for i, col in enumerate(feature_cols) if col in num_cols]], axis=0))
        col_central_type = ['mean'] * len(col_central)
        index = [i for i, col in enumerate(feature_cols) if col in num_cols]
        col_central = col_central + list(stats.mode(X[:, [i for i, col in enumerate(feature_cols) if not col in num_cols]], axis=0).mode[0])
        col_central_type = col_central_type + ['mode']*len(set(feature_cols) - set(num_cols))
        index = index +  [i for i, col in enumerate(feature_cols) if col not in num_cols]
        centroid = pd.DataFrame({'centroid': col_central, 'central_tendency': col_central_type}, index=index)
        centroid.sort_index(inplace=True)
    else:
        centroid = pd.DataFrame({'centroid': np.mean(X, axis=0), 'central_tendency': ['mean']*X.shape[1]})
        
    return centroid


def external_evaluation(X_test, Y_test, X_train, classifier,
                        dimensionality_reduction_method,
                        dimensionality_reduction,
                        classification_type, id):
    '''
    Function for external evaluation on the test dataset
    '''

    dir_path = os.path.dirname(os.path.realpath(__file__)) # current directory path

    # Training data centroid
    centroid_path = f'{dir_path}/output/data_centroid_id_{id}.csv'
    if os.path.isfile(centroid_path):
        centroid = pd.read_csv(centroid_path, index_col=0)
    else:
        feature_cols = pd.read_csv(f'{dir_path}/../data_preparation/output/input_features_id_{id}.csv',
                                    header=None)
        feature_cols = feature_cols.to_dict(orient='list')
        feature_dtype = pd.read_csv(f'{dir_path}/../data_preparation/output/input_features_dtype_{id}.csv',
                                    header=None)
        feature_dtype = feature_dtype.to_dict(orient='list')
        num_cols = [col for col, datatype in feature_dtype.items() if datatype != 'object']
        centroid = centroid_cal(dimensionality_reduction_method,
                                dimensionality_reduction,
                                X_train, feature_cols, num_cols)
        centroid.to_csv(centroid_path)

    # Threshold distance
    distances_train = calc_distance(dimensionality_reduction_method,
                                    dimensionality_reduction,
                                    X_train, centroid)
    q1, q2, q3 = np.quantile(distances_train, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    cut_off_threshold = q2 + 1.5*iqr

    if classification_type == 'multi-label classification':
        loss_metric = 'hamming_loss'
    else:
        loss_metric = 'error'

    # Calculating distances for the test data to the traing data centroid
    distances_test = calc_distance(dimensionality_reduction_method,
                                dimensionality_reduction,
                                X_test, centroid)

    # Predictions on test set
    start_time = time.time()
    Y_test_hat = np.where(classifier.predict(X_test) > 0.5, 1, 0)
    evaluation_time = round(time.time() - start_time, 2)

    # Global evaluation for the model
    global_accuracy = prediction_evaluation(Y_test, Y_test_hat, metric='accuracy')
    global_f1 = prediction_evaluation(Y_test, Y_test_hat, metric='f1')
    global_hamming_loss_or_error = prediction_evaluation(Y_test, Y_test_hat, metric=loss_metric)

    # Evaluation for the model (outside AD)
    outside_ad_accuracy = prediction_evaluation(Y_test[distances_test > cut_off_threshold],
                                            Y_test_hat[distances_test > cut_off_threshold],
                                            metric='accuracy')
    outside_ad_f1 = prediction_evaluation(Y_test[distances_test > cut_off_threshold],
                                            Y_test_hat[distances_test > cut_off_threshold],
                                            metric='f1')
    outside_ad_hamming_loss_or_error = prediction_evaluation(Y_test[distances_test > cut_off_threshold],
                                            Y_test_hat[distances_test > cut_off_threshold],
                                            metric=loss_metric)

    # Evaluation for the model (inside AD)
    inside_ad_accuracy = prediction_evaluation(Y_test[distances_test <= cut_off_threshold],
                                            Y_test_hat[distances_test <= cut_off_threshold],
                                            metric='accuracy')
    inside_ad_f1 = prediction_evaluation(Y_test[distances_test <= cut_off_threshold],
                                        Y_test_hat[distances_test <= cut_off_threshold],
                                        metric='f1')
    inside_ad_hamming_loss_or_error = prediction_evaluation(Y_test[distances_test <= cut_off_threshold],
                                        Y_test_hat[distances_test <= cut_off_threshold],
                                        metric=loss_metric)

    n_outside = Y_test[distances_test > cut_off_threshold].shape[0]
    n_inside = Y_test[distances_test <= cut_off_threshold].shape[0]

    external_testing_results = {
        'global_accuracy': global_accuracy,
        'global_f1': global_f1,
        'global_hamming_loss_or_error': global_hamming_loss_or_error,
        'outside_ad_accuracy': outside_ad_accuracy,
        'outside_ad_f1': outside_ad_f1,
        'outside_ad_hamming_loss_or_error': outside_ad_hamming_loss_or_error,
        'inside_ad_accuracy': inside_ad_accuracy,
        'inside_ad_f1': inside_ad_f1,
        'inside_ad_hamming_loss_or_error': inside_ad_hamming_loss_or_error
    }


    return external_testing_results, evaluation_time, cut_off_threshold, n_outside, n_inside

