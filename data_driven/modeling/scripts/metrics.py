#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importing libraries
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import numpy as np


def accuracy(y_true, y_pred, thresh=0.5):
    '''
    Function to calculate the accuracy/hamming score for multilabel classification
    '''

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater(y_pred, thresh), tf.float32)
    intersection = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32), axis=1)
    union = tf.reduce_sum(tf.cast((y_true == 1) | (y_pred == 1), tf.float32), axis=1)
    division = intersection/union
    division = tf.where(tf.math.is_nan(division), 1.0, division)
    result = tf.reduce_mean(division)
    
    return result


def prediction_evaluation(Y_true, Y_pred, metric='accuracy'):
    '''
    Function to assess the final model
    '''

    if metric == 'accuracy':
        if len(Y_true.shape) == 1:
            return accuracy_score(Y_true, Y_pred)
        else:
            return accuracy(Y_true, Y_pred)
    elif metric == 'f1':
        if len(Y_true.shape) == 1:
            if len(np.unique(Y_true)) == 2:
                return f1_score(Y_true, Y_pred, average='binary', zero_division=0)
            else:
                return f1_score(Y_true, Y_pred, average='macro', zero_division=0)
        else:
            return f1_score(Y_true, Y_pred, average='samples', zero_division=0)
    elif metric == '0_1_loss':
        return 1 - accuracy_score(Y_true, Y_pred, normalize=True, sample_weight=None)
    elif metric == 'hamming_loss':
        return hamming_loss(Y_true, Y_pred)
    elif metric == 'error':
        return 1 - accuracy_score(Y_true, Y_pred)


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