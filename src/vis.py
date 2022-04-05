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
        
        
def plot_directions(y_true, y_preds):
    x = np.array(range(1, 20, 1))
    y = np.array([1 for i in range(20)])

    plt.xlim(-1, 20)
    plt.ylim(0, 5)
    
    for city_idx in range(y_true.shape[1]):
        for i in range(0,len(x)):
            if i == len(x)-1:
                draw_line(x[i],y[i],y_preds[i, 0],2,color='g', label='Prediction')
                draw_line(x[i],y[i],y_true[i, 0],2, color='r', label='Ground Truth')
            else:
                draw_line(x[i],y[i],y_preds[i, 0],2,color='g')
                draw_line(x[i],y[i],y_true[i, 0],2, color='r')
        plt.title('City '+str(city_idx))
        plt.legend()
        plt.show()


def draw_line(x,y,angle,length, color, label=False):
    cartesianAngleRadians = (450-angle)*np.pi/180.0
    terminus_x = x + length * np.cos(cartesianAngleRadians)
    terminus_y = y + length * np.sin(cartesianAngleRadians)
    if label:
        plt.plot([x, terminus_x],[y,terminus_y], linewidth=2, color=color, label=label)
    else:
        plt.plot([x, terminus_x],[y,terminus_y], linewidth=2, color=color)
        
    plt.tick_params(axis='both', which='both', bottom=False, top=False, 
                left=False,right=False, labelleft=False, labelbottom=False)
    plt.xlabel('Time')
    

