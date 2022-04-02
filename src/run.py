#!/usr/bin/env python3

__author__ = 'Mohamed Radwan'


import numpy as np
import torch
from torch import nn

from tqdm import tqdm

from .data_utils import convert2deg

def specs(model):
    lr = 0.001
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return loss_fn, optimizer


def validate(model, val_iter, loss_fn, features_set=2):
    model.eval()
    losses = []
        
    with torch.no_grad():
        for data_batch in tqdm(val_iter):
            if features_set == 1:
                x, y = data_batch
                outputs = model(x)
            else:
                x1, x2, y = data_batch
                outputs = model(x1, x2)

            output_idx = -1
            loss = loss_fn(outputs[output_idx].cpu(), y.cpu())
            losses.append(loss)

    mean_loss = np.mean(losses)
    return mean_loss


def run_train(model, train_iter, val_iter, num_epochs=10, features_set=2, 
        outputs_nr=1):
    loss_fn, optimizer = specs(model)
    
    for epoch in range(num_epochs):
        losses = []
        for data_batch in tqdm(train_iter):
            if features_set == 1:
                x, y = data_batch
                outputs = model(x)
            else:
                x1, x2, y = data_batch
                outputs = model(x1, x2)
            
            optimizer.zero_grad()
            
            if outputs_nr == 1:
                loss = loss_fn(outputs, y)
                loss.backward()
                losses.append(loss.item())

            #handle mulitple losses/outputs
            else:
                for loss_idx in range(outputs_nr):
                    loss = loss_fn(outputs[loss_idx], data_batch[loss_idx])
                    loss.backward(retain_graph=True)
                    # we backward all losses, but only need to show "Y" losses
                    if (loss_idx - outputs_nr) == -1:
                        losses.append(loss.item())
            
            optimizer.step()
        
        train_loss = np.mean(losses)
        val_loss = validate(model, val_iter, loss_fn, 
                features_set=features_set) 
    
        if epoch % 2 == 0:
            print('Epoch: ', epoch+1, ', Train Loss: ', 
                    train_loss, ', Val Loss: ', val_loss)

    return model


def run_test(model, test_iter, scaler, features_set=2, outputs_nr=1):
    model.eval()
    y_preds = list()
    y_true = list()

    max_wind = scaler['feature_max_train'][0]
    min_wind = scaler['feature_min_train'][0]

    with torch.no_grad():
        for data_batch in tqdm(test_iter):
            if features_set == 1:
                x, y = data_batch
                y = y.cpu().numpy().reshape(-1)

                if outputs_nr == 1:
                    y_pred = model(x).view(len(y), -1).cpu().numpy().reshape(-1)
                else:
                    y_pred = model(x)[-1].view(len(y), -1).cpu().numpy().reshape(-1)
            else:
                x1, x2, y = data_batch
                y = y.cpu().numpy().reshape(-1)
                if outputs_nr == 1:
                    y_pred = model(x1, x2).view(len(y), -1).cpu().numpy().reshape(-1)
                else:
                    y_pred = model(x1, x2)[-1].view(len(y), -1).cpu().numpy().reshape(-1)
                
            y = y * max_wind + min_wind
            y_pred = y_pred * max_wind + min_wind
            y_preds.extend(list(y_pred))
            y_true.extend(list(y))
        
    y_preds = np.array(y_preds)
    y_true = np.array(y_true)
    y_true = y_true.reshape(int(y_true.shape[0]/7), 7)
    y_preds = y_preds.reshape(int(y_preds.shape[0]/7), 7)
    
    return y_true, y_preds


def run_test_direction(model, test_iter, scaler, features_set=2, outputs_nr=1, 
        output_sine=False):
    model.eval()
    y_preds = list()
    y_true = list()

    with torch.no_grad():
        for data_batch in test_iter:
            if features_set == 1:
                x, y = data_batch
                if outputs_nr == 1:
                    y_pred = model(x).view(len(y), -1).cpu().numpy().reshape(-1)
                else:
                    y_pred = model(x)[-1].view(len(y), -1).cpu().numpy().reshape(-1)
            else:
                x1, x2, y = data_batch
                if outputs_nr == 1:
                    y_pred = model(x1, x2).view(len(y), -1).cpu().numpy().reshape(-1)
                else:
                    y_pred = model(x1, x2)[-1].view(len(y), -1).cpu().numpy().reshape(-1)
            y = y.cpu().numpy().reshape(-1)
            
            """One drawback in this model is that the values has to 
            be between -1 and 1 in order to find the arcsine of the 
            outputs. So, we round values outside range(-1, 1)"""
            y = np.array(y)
            if not output_sine:
                y_pred = np.array([-1 if i < -1 else i for i in y_pred]) 
                y_pred = np.array([1  if i > 1 else i for i in y_pred])
    
                """split cosine and sine values
                and add very small values (1e-18) to avoid zeros in the matrices"""
                sin_pred = y_pred.reshape(int(y_pred.shape[0]/14), 14)[:, :7].flatten() + 1e-18
                cos_pred = y_pred.reshape(int(y_pred.shape[0]/14), 14)[:, 7:].flatten() + 1e-18
                sin_y = y.reshape(int(y.shape[0]/14), 14)[:, :7].flatten() + 1e-18
                cos_y = y.reshape(int(y.shape[0]/14), 14)[:, 7:].flatten() + 1e-18
            
                # recover the radians from the sine and cosine values
                y_pred, _  = convert2deg(sin_pred,cos_pred)
                y, _  = convert2deg(sin_y,cos_y)
            
            else:
                y_pred = y_pred.reshape(int(y_pred.shape[0]/14), 14)[:, :7].flatten()
                y = y.reshape(int(y.shape[0]/14), 14)[:, :7].flatten() 

            y_preds.extend(list(y_pred))
            y_true.extend(list(y))
        
    y_preds = np.array(y_preds)
    y_true = np.array(y_true)
    y_true = y_true.reshape(int(y_true.shape[0]/7), 7)
    y_preds = y_preds.reshape(int(y_preds.shape[0]/7), 7)
    
    return y_true, y_preds
