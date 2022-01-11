#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import numpy as np


def accuracy(y_true, y_pred, thresh=0.5):
    '''
    Function to calculate the accuracy/hamming score score for multilabel classification
    '''

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater(y_pred, thresh), tf.float32)
    intersection = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32), axis=1)
    union = tf.reduce_sum(tf.cast((y_true == 1) | (y_pred == 1), tf.float32), axis=1)
    division = intersection/union
    division = tf.where(tf.math.is_nan(division), 1.0, division)
    result = tf.reduce_mean(division)
    
    return result


def prediction_evaluation(classifier=None, X=None, Y=None, Y_pred=None, metric='accuracy'):
    '''
    Function to assess the final model
    '''
    
    if classifier:
        Y_pred = np.where(classifier.predict(X) > 0.5, 1, 0)

    if metric == 'accuracy':
        if Y.shape[1] == 1:
            return round(accuracy_score(Y, Y_pred), 2)
        else:
            return round(accuracy(Y, Y_pred), 2)
    elif metric == 'f1':
        if Y.shape[1] == 1:
            if len(np.unique(Y[:, 0])) == 2:
                return round(f1_score(Y, Y_pred, average='binary'), 2)
            else:
                return round(f1_score(Y, Y_pred, average='micro'), 2)
        else:
            return round(f1_score(Y, Y_pred, average='samples'), 2)
    elif metric == '0_1_loss':
        return round(1 - accuracy_score(Y, Y_pred, normalize=True, sample_weight=None), 2)
    elif metric == 'hamming_loss':
        return round(hamming_loss(Y, Y_pred), 2)
    elif metric == 'error':
        return round(1 - accuracy_score(Y, Y_pred), 2)


def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    
    return macro_cost