#!/usr/bin/env python3

__author__ = 'Mohamed Radwan'

import pickle
import numpy as np
import math


def read_raw_data(data_dir='Wind_data_NL'):
    with open(data_dir+'/dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    with open(data_dir+'/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    train_data = data['train']
    test_data = data['test']
    return train_data, test_data, scaler


def build_data(data, x_len, y_len, gap=1):
    x = []
    y = []
    length = data.shape[0]
    for end_idx in range(x_len + y_len + gap, length):
        xtime = data[end_idx-y_len-x_len-gap:end_idx-y_len-gap]
        ytime = data[end_idx-y_len:end_idx]
        x.append(xtime)
        y.append(ytime)
    x = np.stack(x)
    y = np.stack(y)
    return x, y


def make_ready_data(data, train=True, feature='speed', gap=1):

    x_len = 10  # 10 historical time steps
    y_len = 1  # next step

    """we need the tempretaure in addition to the wind speed"""
    if feature == 'tempreture':
        idx = 2
    elif feature == 'speed':
        idx = 0
    elif feature == 'direction':
        idx = 1
        
    
    x, y = build_data(data[:, :, idx], x_len, y_len, gap)
    x, y = x.reshape(x.shape[0], 10, 7), y.reshape(y.shape[0], 7)

    if train:
        xtrain = x[:60000]
        ytrain = y[:60000]
        xval = x[60000:]
        yval = y[60000:]

        return xtrain, xval, ytrain, yval
    else:
        return x, y

    
def make_wind_direction_data(data, train=True, temp=False, gap=1):
    """The direction data is slightly differnet from the speed data.
    Here, we calculate the sine and cosine of each data point. This
    means that the data features will be of the size 14 instead of 7"""
    
    x_len = 10  # 10 historical time steps
    y_len = 1  # next step
    
    if temp:
        idx = 4
    else:
        idx = 1
        
    x, y = build_data(data[:, :, idx], x_len, y_len, gap) 
    
    """Convert the degrees into radians values and take the sine or cosine"""
    x = x.reshape(x.shape[0], 10, 7) * 2 * math.pi
    y = y.reshape(y.shape[0], 7) *  2 * math.pi
    
    xcos, ycos = np.cos(x), np.cos(y)
    xsin, ysin = np.sin(x), np.sin(y)
    
    x = np.concatenate((xsin, xcos), axis=2)
    y = np.concatenate((ysin, ycos), axis=1)
    
    if train:
        xtrain = x[:60000]
        ytrain = y[:60000]
        xval = x[60000:]
        yval = y[60000:]
    
        return xtrain, xval, ytrain, yval
    else:
        return x, y