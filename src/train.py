#!/usr/bin/env python3

__author__ = 'Mohamed Radwan'


from .load_data import read_raw_data, make_ready_data, make_wind_direction_data
from .data_utils import build_dataloader
from .models import LSTM, BiLinearPoolingLSTM, BiLinearPoolingAutoEncoderLSTM
from .run import run_train, run_test, run_test_direction
from .vis import results


def train_speed_data(num_epochs=10):
    #read data
    print('Reading Data...')
    train_data, test_data, scaler = read_raw_data()
    # input and output size are the number of series (cities)
    output_size = 7
    input_size = [7, 7]
    hidden_size = 16
    num_layers = 1
    hours_ahead = 10 #1 Hour ahead, change to (2, 5, 10, 20...etc)

    print('Hours ahead: '+str(hours_ahead)+'H')
    print('Training LSTM model...')
    xtrain, xval, ytrain, yval = make_ready_data(train_data, feature='speed',
        gap=hours_ahead)
    xtest, ytest = make_ready_data(test_data, train=False, feature='speed', 
        gap=hours_ahead)
    train_iter, val_iter, test_iter, device = build_dataloader(xtrain, xval, 
        xtest, ytrain, yval, ytest)

    lstm_baseline = LSTM(output_size, input_size, hidden_size, num_layers)
    lstm_baseline = lstm_baseline.to(device)
    
    lstm_baseline = run_train(lstm_baseline, train_iter, val_iter, num_epochs=num_epochs, 
        features_set=1)
    y_true, y_preds = run_test(lstm_baseline, test_iter, scaler, features_set=1)
    
    print('Error Metrics:')
    results(y_true, y_preds,feature_name = 'speed', plots=False)

    
    print('Training BiLinear LSTM model...')
    xtrain, xval, ytrain, yval = make_ready_data(train_data, 
        feature='speed',gap=hours_ahead)
    xtrain_temp, xval_temp, _, _ = make_ready_data(train_data, 
        feature='temperature', gap=hours_ahead)
    xtest, ytest = make_ready_data(test_data, train=False, 
        feature='speed', gap=hours_ahead)
    xtest_temp, _ = make_ready_data(test_data, train=False, 
        feature='temperature', gap=hours_ahead)
    train_iter, val_iter, test_iter, device = build_dataloader(xtrain, xval, 
                                                           xtest, ytrain, 
                                                           yval, ytest,
                                                           xtrain_temp, 
                                                           xval_temp,
                                                           xtest_temp, 
                                                           add_temp=True)

    lstm_model = BiLinearPoolingLSTM(output_size, input_size, hidden_size, 
        num_layers)
    lstm_model = lstm_model.to(device)
    lstm_model = run_train(lstm_model, train_iter, val_iter, num_epochs=num_epochs)
    y_true, y_preds = run_test(lstm_model, test_iter, scaler)

    print('Error Metrics:')
    results(y_true, y_preds,feature_name = 'speed', plots=False)
    
    print('Training AutoLSTM...')
    autoenc_lstm = BiLinearPoolingAutoEncoderLSTM(output_size, input_size, 
        hidden_size, num_layers)
    autoenc_lstm = autoenc_lstm.to(device)
    autoenc_lstm = run_train(autoenc_lstm, train_iter, val_iter, num_epochs=num_epochs, 
        outputs_nr=3)
    y_true, y_preds = run_test(autoenc_lstm, test_iter, scaler, features_set=2, outputs_nr=3)
    print('Error Metrics:')
    results(y_true, y_preds, feature_name = 'speed', plots=False)


def train_direction_data(num_epochs=10):
    print('Reading Data...')
    train_data, test_data, scaler = read_raw_data()
    hours_ahead = 10
    
    print('Hours ahead: '+str(hours_ahead)+'H')
    xtrain, xval, ytrain, yval = make_wind_direction_data(train_data, 
        gap=hours_ahead)
    xtest, ytest = make_wind_direction_data(test_data, train=False, 
        gap=hours_ahead)
    train_iter, val_iter, test_iter, device = build_dataloader(xtrain, xval, 
        xtest, ytrain, yval, ytest)

    """input and output size are the number of series (cities) twice
    because we ave cosine and sine direction"""
    input_size = [14, 14]
    output_size = 14
    hidden_size = 16
    num_layers = 1


    print('Training LSTM...')
    lstm_model = LSTM(output_size, input_size, hidden_size, num_layers)
    lstm_model = lstm_model.to(device)
    lstm_model = run_train(lstm_model, train_iter, val_iter, num_epochs=num_epochs,
            features_set=1)
    y_true, y_preds = run_test_direction(lstm_model, test_iter, scaler, 
            features_set=1, output_sine=True)

    print('Error Metrics:')
    results(y_true, y_preds, feature_name = 'direction', plots=False)

    xtrain_temp, xval_temp, _, _ = make_ready_data(train_data, 
            feature='temperature', gap=hours_ahead)
    xtest_temp, _ = make_ready_data(test_data, train=False, 
            feature='temperature', gap=hours_ahead)
    xtrain, xval, ytrain, yval = make_wind_direction_data(train_data, 
            gap=hours_ahead)
    xtest, ytest = make_wind_direction_data(test_data, train=False, 
            gap=hours_ahead)
    train_iter, val_iter, test_iter, device = build_dataloader(xtrain, xval, 
            xtest, ytrain, yval, ytest, xtrain_temp, xval_temp, xtest_temp, 
            add_temp=True)
    
    print('Training BiLinear LSTM...')
    input_size = [14, 7] # cosine+sine features = 14, tempretaure features = 7
    output_size = 14 # cosine+sine features = 14
    lstm_model = BiLinearPoolingLSTM(output_size, input_size, hidden_size, 
            num_layers)
    lstm_model = lstm_model.to(device)
    lstm_model = run_train(lstm_model, train_iter, val_iter, num_epochs=num_epochs)
    y_true, y_preds = run_test_direction(lstm_model, test_iter, scaler, 
            output_sine=True)

    print('Error Metrics:')
    results(y_true, y_preds, feature_name = 'direction', plots=False)

    print('Training AutoLSTM...')
    model = BiLinearPoolingAutoEncoderLSTM(output_size, input_size, hidden_size, 
            num_layers)
    model = model.to(device)
    model = run_train(model, train_iter, val_iter, num_epochs=num_epochs, outputs_nr=3)
    y_true, y_preds = run_test_direction(model, test_iter, scaler, outputs_nr=3, 
            output_sine=True)

    print('Error Metrics:')
    results(y_true, y_preds, feature_name = 'direction', plots=False)

