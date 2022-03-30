#!/usr/bin/env python3

__author__ = 'Mohamed Radwan'


import torch
from torch import nn

from tqdm import tqdm

def specs(model):
    lr = 0.01
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return loss_fn, optimizer


def validate(model, val_iter):
    model.eval()
    loss_fn, optimizer = specs(model)
    losses = []
    with torch.no_grad():
        for x1, x2, y in tqdm(val_iter):
            outputs = model(x1, x2)
            loss = loss_fn(outputs.cpu(), y.cpu())
            losses.append(loss)
    mean_loss = np.mean(losses)
    return mean_loss


def run_train(model, train_iter, val_iter, num_epochs=10):
    loss_fn, optimizer = specs(model)
    for epoch in range(num_epochs):
        losses = []
        #
        for x1, x2, y in tqdm(train_iter):
            outputs = model(x1, x2)
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        train_loss = np.mean(losses)
        val_loss = validate(model, val_iter) 
    
        if epoch % 2 == 0:
            print('Epoch: ', epoch+1, ', Train Loss: ', train_loss, ', Val Loss: ', val_loss)
                  
    return model


def run_test(model, test_iter, scaler):
    model.eval()
    y_preds = list()
    y_true = list()

    max_wind = scaler['feature_max_train'][0]
    min_wind = scaler['feature_min_train'][0]

    with torch.no_grad():
        for x1, x2, y in tqdm(test_iter):
            y = y.cpu().numpy().reshape(-1)
            y_pred = model(x1, x2).view(len(y), -1).cpu().numpy().reshape(-1)
            y = y * max_wind + min_wind
            y_pred = y_pred * max_wind + min_wind
            y_preds.extend(list(y_pred))
            y_true.extend(list(y))
        
    y_preds = np.array(y_preds)
    y_true = np.array(y_true)
    y_true = y_true.reshape(int(y_true.shape[0]/7), 7)
    y_preds = y_preds.reshape(int(y_preds.shape[0]/7), 7)
    
    return y_true, y_preds

