#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
import numpy as np
from tqdm import tqdm
from scipy.stats import mannwhitneyu

def stratified_k_fold_cv(classifier, X, Y):
    '''
    Function to apply stratified k-fold cross validation
    '''

    skfold = StratifiedKFold(n_splits=5,
                            random_state=100,
                            shuffle=True)

    results_skfold = cross_validate(classifier,
                                    X, Y,
                                    cv=skfold,
                                    n_jobs=4,
                                    scoring=('balanced_accuracy',
                                            'accuracy'),
                                    return_train_score=True)
    
    balanced_accuracy_validation = round(np.mean(results_skfold['test_balanced_accuracy']), 2)
    balanced_accuracy_train = round(np.mean(results_skfold['train_balanced_accuracy']), 2)
    balanced_accuracy_analysis = overfitting_underfitting(
                                        results_skfold['train_balanced_accuracy'],
                                        results_skfold['test_balanced_accuracy']
                                        )
    error_train = round(np.mean(1 - results_skfold['train_accuracy']), 2)
    error_validation = round(np.mean(1 - results_skfold['test_accuracy']), 2)
    std_error_train = round(np.std(1 - results_skfold['train_accuracy']), 6)
    std_error_validation = round(np.std(1 - results_skfold['test_accuracy']), 6)

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

    if np.mean(score_train) < 0.75:
        U1, p = mannwhitneyu(score_train,
                            score_test,
                            method="exact")
        if p < 0.05:
            return 'under-fitting (high bias and high variance)'
        else:
            return 'under-fitting (high bias)'
    else:
        U1, p = mannwhitneyu(score_train,
                            score_test,
                            method="exact")
        if p < 0.05:
            return 'over-fitting (high variance)'
        else:
            return 'optimal-fitting'


def y_randomization(classifier, X, Y):
    '''
    Function to apply Y-Randomization
    '''

    shuffled_errors = []
    indexes = Y.index.tolist()
    pbar = tqdm(total=30)
  
    for i in range(30):
        np.random.shuffle(indexes)
        
        classifier.fit(X,Y[indexes])
        
        Ypred = classifier.predict(X)
        shuffled_errors.append(1 - accuracy_score(Y[indexes],Ypred))

        pbar.update(n=1)
    
    
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
    elif metric == 'matrix':
        return confusion_matrix(Y, Y_pred)
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
    '''

    # Criterion 1
    df['criterion_1'] = np.abs(df[17] - df[18])
    df['criterion_1'] = (df['criterion_1'] - df['criterion_1'].max())/(0 - df['criterion_1'].max())

    # Criterion 2
    df['criterion_2'] = df[17]

    # Criterion 3
    df.loc[df[27] == 0, 27] = 10 ** -4
    df['criterion_3'] = df[26]/df[27]
    df['criterion_3'] = (df['criterion_3'] - df['criterion_3'].max())/(df['criterion_3'].min() - df['criterion_3'].max())
    
    # Criterion 4
    df['criterion_4'] = (df[24] - df[25]) - df[20]
    df['criterion_4'] = (df['criterion_4'] - df['criterion_4'].min())/(df['criterion_4'].max() - df['criterion_4'].min())

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


def parameter_tuning(X, Y, model, fixed_params, space):
    '''
    Function to search parameters based on randomized grid search
    '''

    skfold = StratifiedKFold(n_splits=5,
                            random_state=100,
                            shuffle=True)

    classifier = DataDrivenModel(model, **fixed_params)

    search = RandomizedSearchCV(classifier, space,
                        n_iter=100, 
                        scoring=('balanced_accuracy',
                                'accuracy'),
                        n_jobs=-1, cv=skfold,
                        random_state=1,
                        return_train_score=True,
                        refit=True)

    search.fit(X, Y)

    results = search.cv_results_
    time = search.refit_time_
    best_estimator = search.best_estimator_

    return results, time, best_estimator