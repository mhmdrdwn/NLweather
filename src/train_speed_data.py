#!/usr/bin/env python3

__author__ = 'Mohamed Radwan'


from src.load_data import read_raw_data, make_ready_data
from src.data_utils import build_dataloader
from src.models import LSTM, BiLinearPoolingLSTM, BiLinearPoolingAutoEncoderLSTM
from src.run import run_train, validate
from src.run import run_test
from src.vis import results

#read data
train_data, test_data, scaler = read_raw_data()
# input and output size are the number of series (cities)
output_size = 7
input_size = [7, 7]
# hidden size is number of LSTM units, this means 32 as it's BiLSTM
hidden_size = 16
# Only one layer of LSTM
num_layers = 1

# build time series features and labels
hours_ahead = 1 #1 Hour ahead, This is to be changed to (5, 10, 50...etc)
xtrain, xval, ytrain, yval = make_ready_data(train_data, feature='speed',gap=hours_ahead)
xtest, ytest = make_ready_data(test_data, train=False, feature='speed', gap=hours_ahead)
train_iter, val_iter, test_iter, device = build_dataloader(xtrain, xval, xtest, ytrain, yval, ytest)

# build the model
lstm_baseline = LSTM(output_size, input_size, hidden_size, num_layers)
lstm_baseline = lstm_baseline.to(device)
# train the model
lstm_baseline = run_train(lstm_baseline, train_iter, val_iter, num_epochs=10, features_set=1)
#now we test the model
y_true, y_preds = run_test(lstm_baseline, test_iter, scaler, features_set=1)

#print results
results(y_true, y_preds,feature_name = 'speed', plots=False)


# build time series features and labels including temperature time series
xtrain, xval, ytrain, yval = make_ready_data(train_data, feature='speed',gap=hours_ahead)
xtrain_temp, xval_temp, _, _ = make_ready_data(train_data, feature='temperature', gap=hours_ahead)
xtest, ytest = make_ready_data(test_data, train=False, feature='speed', gap=hours_ahead)
xtest_temp, _ = make_ready_data(test_data, train=False, feature='temperature', gap=hours_ahead)
train_iter, val_iter, test_iter, device = build_dataloader(xtrain, xval, xtest,
                                                           ytrain, yval, ytest,
                                                           xtrain_temp, xval_temp,
                                                           xtest_temp, add_temp=True)

#build BiLinear LSTM model and train
lstm_model = BiLinearPoolingLSTM(output_size, input_size, hidden_size, num_layers)
lstm_model = lstm_model.to(device)
lstm_model = run_train(lstm_model, train_iter, val_iter, num_epochs=10)
y_true, y_preds = run_test(lstm_model, test_iter, scaler)
results(y_true, y_preds,feature_name = 'speed', plots=False)

#build autencoderLSTM model and train
autoenc_lstm = BiLinearPoolingAutoEncoderLSTM(output_size, input_size, hidden_size, num_layers)
autoenc_lstm = autoenc_lstm.to(device)
autoenc_lstm = run_train(autoenc_lstm, train_iter, val_iter, num_epochs=10, outputs_nr=3)
y_true, y_preds = run_test(autoenc_lstm, test_iter, scaler, features_set=2)
results(y_true, y_preds, feature_name = 'speed', plots=False)

