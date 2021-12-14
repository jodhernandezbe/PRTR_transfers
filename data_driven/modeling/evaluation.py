#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, make_scorer, r2_score
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from scipy import stats


def performing_cross_validation(regressor, X, Y):
    '''
    Function to apply k-fold cross validation
    '''

    results_kfold = cross_validate(regressor,
                                    X, Y,
                                    cv=5,
                                    n_jobs=-1,
                                    scoring={'mean_absolute_percentage_error': make_scorer(mean_absolute_percentage_error),
                                            'r2': make_scorer(r2_score, multioutput='variance_weighted'),
                                            'mean_squared_error': make_scorer(mean_squared_error)},
                                    return_train_score=True)

    print(results_kfold)
    
    mape_analysis = overfitting_underfitting(
                                        results_kfold['train_mean_absolute_percentage_error'],
                                        results_kfold['test_mean_absolute_percentage_error']
                                        )
    mape_train = round(np.mean(results_kfold['train_mean_absolute_percentage_error']), 2)
    mape_validation = round(np.mean(results_kfold['test_mean_absolute_percentage_error']), 2)
    std_mape_train = round(np.std(results_kfold['train_mean_absolute_percentage_error']), 6)
    std_mape_validation = round(np.std(results_kfold['test_mean_absolute_percentage_error']), 6)

    cv_result = {'mape_validation': mape_validation,
            'mape_train': mape_train,
            'mape_analysis': mape_analysis,
            'std_mape_train': std_mape_train,
            'std_mape_validation': std_mape_validation}

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

    mean_score_test = round(np.mean(score_test), 2)
    U1, p = mannwhitneyu(score_train,
                        score_test,
                        method="exact")

    # Analysing MAPE
    if mean_score_test < 0.1:
        string = 'MAPE is very good'
    elif (mean_score_test >= 0.1) and (mean_score_test < 0.2):
        string = 'MAPE is good'
    elif (mean_score_test >= 0.2) and (mean_score_test < 0.5):
        string = 'MAPE is ok'
    elif mean_score_test >= 0.5:
        string = 'MAPE is not good'

    # Analysing the difference beteween the scores
    if p < 0.05:
        return f'{string} (variance is high)'
    else:
        return f'{string} (variance is not high)'


def y_randomization(regressor, X, Y):
    '''
    Function to apply Y-Randomization
    '''

    shuffled_errors = []
    indexes = list(range(Y.shape[0]))

    def inner_loop():
        np.random.shuffle(indexes)
        regressor.fit(X,Y[indexes])
        Ypred = regressor.predict(X)
        return mean_absolute_percentage_error(Y[indexes],Ypred)

    shuffled_errors = Parallel(n_jobs=-1)(delayed(inner_loop)() for i in range(30))
    
    y_randomization_error = {'mean_y_randomization_mape': round(np.mean(shuffled_errors), 2),
                            'std_y_randomization_mape': round(np.std(shuffled_errors), 6)}

    return y_randomization_error


def prediction_evaluation(regressor, X, Y, metric='mean_absolute_percentage_error'):
    '''
    Function to assess the final model
    '''

    Y_pred = regressor.predict(X)

    if metric == 'mean_absolute_percentage_error':
        return round(mean_absolute_percentage_error(Y, Y_pred), 2)
    elif metric == 'mean_squared_error':
        return round(mean_squared_error(Y, Y_pred), 2)


def data_driven_models_ranking(df):
    '''
    Function to select the data driven model

    Here a best-worst scaling is applied as well
    criterion = (score -  score_worst)/(score - score_best)
    '''

    dict_for_mape = {'MAPE is very good (variance is not high)': 8,
                    'MAPE is very good (variance is high)': 7,
                    'MAPE is good (variance is not high)': 6,
                    'MAPE is good (variance is high)': 5,
                    'MAPE is ok (variance is not high)': 4,
                    'MAPE is ok (variance is high)': 3,
                    'MAPE is not good (variance is not high)': 2,
                    'MAPE is not good (variance is high)': 1
                    }

    # Criterion 1
    df['criterion_1'] = df[14].apply(lambda x: dict_for_mape[x])
    df['criterion_1'] = (df['criterion_1'] - 1)/(8 - 1)

    # Criterion 2
    df['criterion_2'] = df[15]
    df['criterion_2'] = (df['criterion_2'] - 1)/(0 - 1)

    # Criterion 3
    df.loc[df[22] == 0, 22] = 10 ** -4
    df['criterion_3'] = df[21]/df[22]
    df['criterion_3'] = (df['criterion_3'] - df['criterion_3'].max())/(df['criterion_3'].min() - df['criterion_3'].max())
    
    # Criterion 4
    df['criterion_4'] = df[15]/(df[19] - df[20])
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