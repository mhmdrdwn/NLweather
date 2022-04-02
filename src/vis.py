#!/usr/bin/env python3

__author__ = 'Mohamed Radwan'


import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def results(y_true, y_preds, feature_name, plots=True):
    """Visualization of windspeed predictions vs ground truth and
    Overall evaluation the predictions of the test data"""

    print('RMSE: ', np.sqrt(mean_squared_error(y_true.flatten(), 
        y_preds.flatten())))
    print('MAE: ', mean_absolute_error(y_true.flatten(), 
        y_preds.flatten()))

    if plots:
        for i in range(y_preds.shape[1]):
            plt.figure(figsize=(10,7))
            plt.plot(range(len(y_true[:, i])), y_true[:, i], 
                    label='Ground Truth ')
            plt.plot(range(len(y_preds[:, i])), y_preds[:, i], 
                    label='Predictions')
            plt.title('City '+str(i+1))
            plt.xlabel('Time Index')
            plt.ylabel('Wind ' + feature_name)
            plt.legend(loc="upper left")
            plt.show()


def error_acf(y_true, y_preds):
    """Visualize the residuals correlation with 
    lagged version of each other"""
    residuals = y_true - y_preds
    for city_idx in range(y_preds.shape[1]):
        plt.figure(figsize=(10,7))
        acf_plot = plot_acf(residuals[:, city_idx], lags=50, zero=False)
        plt.ylim(-0.25, 0.25)
        plt.ylabel('ACF', fontsize=12)
        plt.xlabel('Lags', fontsize=12)
        plt.show()

