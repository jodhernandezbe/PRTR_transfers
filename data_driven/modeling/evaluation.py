#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sklearn.model_selection import cross_validate
from sklearn.metrics import balanced_accuracy_score, accuracy_score, make_scorer, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from scipy import stats


def performing_cross_validation(classifier, X, Y):
    '''
    Function to apply k-fold cross validation
    '''

    results_kfold = cross_validate(classifier,
                                    X, Y,
                                    cv=5,
                                    n_jobs=-1,
                                    scoring={'balanced_accuracy': make_scorer(balanced_accuracy_score),
                                            'accuracy': make_scorer(accuracy_score)},
                                    return_train_score=True)

    balanced_accuracy_validation = round(np.mean(results_kfold['test_balanced_accuracy']), 2)
    balanced_accuracy_train = round(np.mean(results_kfold['train_balanced_accuracy']), 2)
    balanced_accuracy_analysis = overfitting_underfitting(
                                        results_kfold['train_balanced_accuracy'],
                                        results_kfold['test_balanced_accuracy']
                                        )
    error_train = round(np.mean(1 - results_kfold['train_accuracy']), 2)
    error_validation = round(np.mean(1 - results_kfold['test_accuracy']), 2)
    std_error_train = round(np.std(1 - results_kfold['train_accuracy']), 6)
    std_error_validation = round(np.std(1 - results_kfold['test_accuracy']), 6)

    cv_result = {'balanced_accuracy_validation': balanced_accuracy_validation,
            'balanced_accuracy_train': balanced_accuracy_train,
            'balanced_accuracy_analysis': balanced_accuracy_analysis,
            'error_train': error_train,
            'error_validation': error_validation,
            'std_error_train': std_error_train,
            'std_error_validation': std_error_validation}

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

    if score_train < 0.75:
        if p < 0.05:
            return 'under-fitting (high bias and high variance)'
        else:
            return 'under-fitting (high bias)'
    else:
        if p < 0.05:
            return 'over-fitting (high variance)'
        else:
            return 'optimal-fitting'


def y_randomization(classifier, X, Y):
    '''
    Function to apply Y-Randomization
    '''

    shuffled_errors = []
    indexes = list(range(Y.shape[0]))

    def inner_loop():
        np.random.shuffle(indexes)
        classifier.fit(X,Y[indexes])
        Ypred = classifier.predict(X)
        return 1 - accuracy_score(Y[indexes],Ypred)

    shuffled_errors = Parallel(n_jobs=-1)(delayed(inner_loop)() for i in range(30))
    
    y_randomization_error = {'mean_y_randomization_error': round(np.mean(shuffled_errors), 2),
                            'std_y_randomization_error': round(np.std(shuffled_errors), 6)}

    return y_randomization_error


def prediction_evaluation(classifier, X, Y, metric='balanced_accuracy'):
    '''
    Function to assess the final model
    '''

    Y_pred = classifier.predict(X)

    if metric == 'balanced_accuracy':
        return round(balanced_accuracy_score(Y, Y_pred), 2)
    elif metric == 'accuracy':
        return round(accuracy_score(Y, Y_pred), 2)
    elif metric == 'f1_weighted':
        return round(f1_score(Y, Y_pred, average='weighted'), 2)
    elif metric == 'recall_weighted':
        return round(recall_score(Y, Y_pred,  average='weighted'), 2)
    elif metric == 'precision_weighted':
        return round(precision_score(Y, Y_pred,  average='weighted'), 2)
    elif metric == 'error':
        return round(1 - accuracy_score(Y, Y_pred), 2)


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
    df['criterion_1'] = df[19].apply(lambda x: dict_for_acc[x])
    df['criterion_1'] = (df['criterion_1'] - 1)/(4 - 1)

    # Criterion 2
    df['criterion_2'] = df[17]
    df['criterion_2'] = (df['criterion_2'] - 0)/(1 - 0)

    # Criterion 3
    df.loc[df[27] == 0, 27] = 10 ** -4
    df['criterion_3'] = df[26]/df[27]
    df['criterion_3'] = (df['criterion_3'] - df['criterion_3'].max())/(df['criterion_3'].min() - df['criterion_3'].max())
    
    # Criterion 4
    df['criterion_4'] = df[20]/(df[24] - df[25])
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

    if (not dimensionality_reduction) or (dimensionality_reduction_method != 'PCA'):
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

    if (not dimensionality_reduction) or (dimensionality_reduction_method != 'PCA'):
        # Ahmad & dey’s distance
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