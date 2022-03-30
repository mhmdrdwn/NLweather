#!/usr/bin/env python3

__author__ = 'Mohamed Radwan'


import torch
import numpy as np

def build_dataloader(xtrain, xval, xtest, ytrain,
                     yval, ytest, xtrain_temp=None, 
                     xval_temp=None, xtest_temp=None, 
                     add_temp=False):
    
    batch_size = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    xtrain = torch.Tensor(xtrain).to(device)
    ytrain = torch.Tensor(ytrain).to(device)
    xval = torch.Tensor(xval).to(device)
    yval = torch.Tensor(yval).to(device)
    xtest = torch.Tensor(xtest).to(device)
    ytest = torch.Tensor(ytest).to(device)
    if add_temp:
        xtest_temp = torch.Tensor(xtest_temp).to(device)
        xtrain_temp = torch.Tensor(xtrain_temp).to(device)
        xval_temp = torch.Tensor(xval_temp).to(device)
        train_data = torch.utils.data.TensorDataset(xtrain, xtrain_temp, ytrain)
        val_data = torch.utils.data.TensorDataset(xval, xval_temp, yval)
        test_data = torch.utils.data.TensorDataset(xtest, xtest_temp, ytest)
    else:
        train_data = torch.utils.data.TensorDataset(xtrain, ytrain)
        val_data = torch.utils.data.TensorDataset(xval, yval)
        test_data = torch.utils.data.TensorDataset(xtest, ytest)
        
    test_iter = torch.utils.data.DataLoader(test_data, batch_size)
    val_iter = torch.utils.data.DataLoader(val_data, batch_size)
    train_iter = torch.utils.data.DataLoader(train_data, batch_size)

    return train_iter, val_iter, test_iter, device


def convert2deg(sin_pred,cos_pred):
    """Modified after: https://mattgorb.github.io/wind,
    The idea here is to convert the sine and cosine values
    into radians
    """
    inv_sin=np.degrees(np.arcsin(sin_pred))
    inv_cos=np.degrees(np.arccos(cos_pred))
    radians_sin=[]
    radians_cos=[]
    
    for a,b,c,d in zip(sin_pred, cos_pred, inv_sin, inv_cos):
        if(a>0 and b>0):
            radians_sin.append(c)
            radians_cos.append(d)
        elif(a>0 and b<0):
            radians_sin.append(180-c)
            radians_cos.append(d)
        elif(a<0 and b<0):
            radians_sin.append(180-c)
            radians_cos.append(360-d)
        elif(a<0 and b>0):
            radians_sin.append(360+c)
            radians_cos.append(360-d)
            
    radians_sin=np.array(radians_sin)
    radians_cos=np.array(radians_cos)
    return radians_sin, radians_cos