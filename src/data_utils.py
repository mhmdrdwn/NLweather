#!/usr/bin/env python3

__author__ = 'Mohamed Radwan'


import torch

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

