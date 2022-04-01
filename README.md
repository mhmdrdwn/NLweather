# NL-weather

Here we see how temperature can have effect on wind speed and direction. we build Fusion neural network where temperature and wind time series data are used together in forcasting of wind speed and wind direction. 

### Data

- The dataset contains wind speed, wind direction, temperature, wind pressure, rain amount and Dew Point (6 features). 
- The data was acquired from 7 cities in Netherlands from 2011 to 2020 with 81.000 datapoints. 
- The data is split into training (January 2011 - December 2018) and testing (January 2019 - March 2020)

### Methods
In all methods, we build the features and outputs using sliding window. The features is 10 steps in time while the outputs are the next time step after a gap. The gaps we used here is time a head wher we want to predict the values 1, 5, 10 and 50 hours ahead.  

**Baseline: Vanilla LSTM using only wind speed data or widn direction**

**LSTM + Bilinear Pooling using wind speed/direction + temperature data**
Speed (or direction) and temperature features are feed to separte two LSTM layers followed. The outputs are cross multipled to form a matrix. The idea is to get all possible (exhaustive) multiplications of the two outputs vectors of LSTM layers. This idea is originally from the article [Tensor Fusion Network](https://arxiv.org/abs/1707.07250) for using on multimodal data.

**Autoencoder LSTM + BiLinear Pooling using wind speed/direction + temperature data**
This is just an extenstion of the LSTM + Bipooling. The idea is to make a less noisy representation of the two data (Speed and temperature data). In that case, The speed and temperature ar fed into two separate LSTM layers (encoder) followed by a bottleneck layer and two separate output LSTM layers (decoder). The goal of the encoder-decoder is to reconstuct the speed and temperature features. While the model is training to reconstruct the features, the bottleneck represenatations is cross multiplied. Here we optimize three losses (speed reconstruction loss, temperture reconstruction loss and output prediction loss).


### Results
[Check Demo](https://github.com/mhmdrdwn/NLweather/blob/main/demo.ipynb)

### Wind Speed

#### Error Metrics

| Error | Model                        | 1H ahead | 5H ahead|10H ahead  |50H ahead    |
|-------| ---------------------------- |:--------:|:-------:|:---------:|:-----------:|
| MAE   | LSTM Baseline                |  8.99    |  13.73  |   17.74   |  18.91      |
| MAE   | LSTM+BiLinPooling            |  7.30    |**9.38** |   11.55   |  17.33      | 
| MAE   | AutoencoderLSTM+BiLinPooling | **6.62** |  9.88   | **10.97** |  **16.45**  |
| RMSE  | LSTM Baseline                | 11.84    |  17.77  |   22.18   |  24.19      |
| RMSE  | LSTM+BiLinPooling            | 10.82    |**13.43**|   16.18   |  22.81      |
| RMSE  | AutoencoderLSTM+BiLinPooling |  **9.12**|  13.67  | **15.97** |  **21.85**  |


#### Sample Visualization

| Vanilla LSTM  | LSTM with BiLinear Pooling | AutoencoderLSTM with BiLinear Pooling |
|---------------| ---------------------------|-------------------------------------- |
| ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/lstm_speed.png) | ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/lstm_bi_speed.png) | ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/ae_bi_speed.png) |


### Wind Direction

#### Error Metrics (Sine of Degrees)

| Error | Model                        | 1H ahead  | 5H ahead  |10H ahead  |50H ahead  |
|-------| ---------------------------- |:---------:|:---------:|:---------:|:---------:|
| MAE   | LSTM Baseline                |  0.196    |  0.305    |   0.388   |  0.537    |
| MAE   | LSTM+BiLinPooling            |  0.165    |  0.249    |   0.290   |  0.494    |
| MAE   | AutoencoderLSTM+BiLinPooling |  **0.163**|  **0.238**| **0.286** |  **0.470**|
| RMSE  | LSTM Baseline                |  0.289    |  0.418    |   0.506   |  0.637    |
| RMSE  | LSTM+BiLinPooling            |  **0.241**|  0.347    |   0.407   |  0.599    |
| RMSE  | AutoencoderLSTM+BiLinPooling |  0.244    |  **0.338**| **0.396** |  **0.576**|


#### Error Metrics (Degrees)

**N.B. Degrees can be misleading as directions 0 and 360 are equal, but it is here for visualization**

| Error | Model                        | 1H ahead  | 5H ahead  |10H ahead  |50H ahead    |
|-------| ---------------------------- |:---------:|:---------:|:---------:|:-----------:|
| MAE   | LSTM Baseline                |  29.08    |  43.29    |   54.65   |  69.39      | 
| MAE   | LSTM+BiLinPooling            |  **22.95**|  34.20    |   41.69   |  65.93      | 
| MAE   | AutoencoderLSTM+BiLinPooling |  24.19    |**32.89**  | **39.87** | **65.72**   |
| RMSE  | LSTM Baseline                |  65.07    |  78.30    |   86.88   | **91.49**   |
| RMSE  | LSTM+BiLinPooling            |  **58.74**|  70.59    |   77.69   |  93.05      |
| RMSE  | AutoencoderLSTM+BiLinPooling |  60.17    |**68.56**  | **74.79** |    94.68    |


#### Sample Visualization

| Vanilla LSTM  | LSTM with BiLinear Pooling | AutoencoderLSTM with BiLinear Pooling |
|---------------| -------------------------- | ------------------------------------- |
| ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/lstm_dir.png) | ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/lstm_bi_dir.png) | ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/ae_bi_dir.png) |


### References: Dataset source and project structure
- Dataset from "Trebing, Kevin and Mehrkanoon, Siamak, 2020, Wind speed prediction using multidimensional convolutional neural networks" [Github](https://github.com/HansBambel/multidim_conv)
- Project structure: [Github](https://github.com/ossez-com/python-project-structure-sample)

