from src.load_data import make_wind_direction_data, read_raw_data
from src.data_utils import build_dataloader
from src.models import LSTM, BiLinearPoolingLSTM, BiLinearPoolingAutoEncoderLSTM
from src.run import run_test_direction, run_train
from src.vis import results

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

#build LSTM and train
lstm_model = LSTM(output_size, input_size, hidden_size, num_layers)
lstm_model = lstm_model.to(device)
lstm_model = run_train(lstm_model, train_iter, val_iter, num_epochs=10)
y_true, y_preds = run_test_direction(lstm_model, test_iter, scaler, 
        output_sine=True)
results(y_true, y_preds, feature_name = 'direction', plots=False)

#build fusion data
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

#build LSTM with fusion and train
input_size = [14, 7] # cosine+sine features = 14, tempretaure features = 7
output_size = 14 # cosine+sine features = 14
lstm_model = BiLinearPoolingLSTM(output_size, input_size, hidden_size, 
        num_layers)
lstm_model = lstm_model.to(device)
lstm_model = run_train(lstm_model, train_iter, val_iter, num_epochs=1)
y_true, y_preds = run_test_direction(lstm_model, test_iter, scaler, 
        output_sine=True)
results(y_true, y_preds, feature_name = 'direction', plots=False)

#build AutoencoderLSTM and train
model = BiLinearPoolingAutoEncoderLSTM(output_size, input_size, hidden_size, 
        num_layers)
model = model.to(device)
model = run_train(model, train_iter, val_iter, num_epochs=10, outputs_nr=3)
y_true, y_preds = run_test_direction(model, test_iter, scaler, outputs_nr=3, 
        output_sine=True)
results(y_true, y_preds, feature_name = 'direction', plots=False)


