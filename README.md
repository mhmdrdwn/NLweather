# NL-weather

Here we build Fusion neural network where temperature and wind time series data are used together in forcasting of wind speed and wind direction. 

### Data
- The dataset was aacuired by the Netherlands Meteorological Institute.
- The dataset contains wind speed, wind direction, temperature, wind pressure, rain amount and Dew Point (6 data features). 
- The data was acquired from 7 cities in Netherlands from 2011 to 2020 with 81.000 datapoints. 
- The data is split into training (January 2011 - December 2018) and testing (January 2019 - March 2020).

#### Related Interesting Studies on the same data

- [Multidimensional convolutional neuralnetworks](https://github.com/HansBambel/multidim_conv)
- [Spatiotemporal graph convolutional neuralnetworks](https://github.com/tstanczyk95/WeatherGCNet)

### Methods
In all methods, we build the features and outputs using sliding window. The features are 10 steps in time while the outputs are the next time step after a gap (lag). The gaps we used here is time a head where we want to predict the values 1, 2, 5 and 10 hours ahead. 

**Baseline: Vanilla LSTM using only wind speed data or wind direction**

**LSTM + Bilinear Pooling using wind speed/direction + temperature data**:
Speed (or direction) and temperature features are feed to separte two LSTM layers. The outputs are cross multipled to form a matrix. The idea is to get all possible (exhaustive) multiplications of the two outputs vectors of LSTM layers. This idea is originally from the article [Tensor Fusion Network](https://arxiv.org/abs/1707.07250) for using on multimodal data.

**Autoencoder LSTM + BiLinear Pooling using wind speed/direction + temperature data**:
This is just an extenstion of the LSTM + Bipooling. The idea is to make a less noisy representation of the two data (Speed and temperature data). In that case, The speed and temperature ar fed into two separate LSTM layers (encoder) followed by a bottleneck layer and two separate output LSTM layers (decoder). The goal of the encoder-decoder is to reconstuct the speed and temperature features. While the model is training to reconstruct the features, the bottleneck represenatations is cross multiplied and mapped to the prediction outputs. Here we optimize three losses (speed reconstruction loss, temperature reconstruction loss and output prediction loss).


### Results
[Check Demo](https://github.com/mhmdrdwn/NLweather/blob/main/demo.ipynb)

Run the experiment using the command (change number of epochs in the main.py file):
```
python main.py
```

### Wind Speed

#### Error Metrics

| Error | Model                        | 1H ahead |2H ahead  | 5H ahead|10H ahead  |
|-------| ---------------------------- |:--------:|:--------:|:-------:|:---------:|
| MAE   | LSTM Baseline                |  8.86    | 10.26    |  13.72  |   16.36   | 
| MAE   | LSTM+BiLinPooling            |  **5.73**| **6.69** |**8.80** |  **10.72**| 
| MAE   | AutoencoderLSTM+BiLinPooling |  6.74    | 7.17     |  9.88   |   11.09   | 


| Error | Model                        | 1H ahead |2H ahead  | 5H ahead|10H ahead  |
|-------| ---------------------------- |:--------:|:--------:|:-------:|:---------:|
| RMSE  | LSTM Baseline                |  11.72   | 13.49    |  17.67  |   20.95   | 
| RMSE  | LSTM+BiLinPooling            |  **9.11**| **10.42**|**13.04**| **15.42** |  
| RMSE  | AutoencoderLSTM+BiLinPooling |  9.54    |  10.46   |  13.67  |   15.62   | 


#### Error Diagnostics for LSTM+BiLinPooling using wind speed data
| City 1  | City 5 | City 7 |
|---------------| ---------------------------|-------------------------------------- |
| ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/city1_error.png) | ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/city5_error.png) | ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/city7_error.png) |

**The ACF of the errors suggests that there is still a little patterns in the test residuals that should have been captured by the model or another model. The residuals still show some time lag correlations. This means the model can still be optimized. This pattern was hoped to be captured using the AutoencoderLSTM but it does look that the AutoencoderLSTM does not give best results in the case study here (But it is has still superior performance than vanilla LSTM)**

#### Sample Visualization

| Vanilla LSTM  | LSTM with BiLinear Pooling | AutoencoderLSTM with BiLinear Pooling |
|---------------| ---------------------------|-------------------------------------- |
| ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/lstm_speed.png) | ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/lstm_bi_speed.png) | ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/ae_bi_speed.png) |


### Wind Direction

#### Error Metrics (Sine of Degrees)

| Error | Model                        | 1H ahead  | 2H ahead | 5H ahead  |10H ahead  |
|-------| ---------------------------- |:---------:|:--------:|:---------:|:---------:|
| MAE   | LSTM Baseline                |  0.196    | 0.222    |  0.307    |   0.391   | 
| MAE   | LSTM+BiLinPooling            |  **0.133**| **0.165**|  **0.222**| **0.278** | 
| MAE   | AutoencoderLSTM+BiLinPooling |    0.154  | 0.169    | 0.238     |   0.286   | 


| Error | Model                        | 1H ahead  | 2H ahead | 5H ahead  |10H ahead  |
|-------| ---------------------------- |:---------:|:--------:|:---------:|:---------:|
| RMSE  | LSTM Baseline                |  0.289    | 0.318    |  0.407    |   0.498   | 
| RMSE  | LSTM+BiLinPooling            |  **0.223**| **0.258**|  **0.325**| **0.392** | 
| RMSE  | AutoencoderLSTM+BiLinPooling |  0.235    | 0.259    |   0.338   |    0.396  | 


#### Error Metrics (Degrees)

**N.B. Degrees can be misleading as directions 0 and 360 are equal, but it is here for visualization**

#### Sample Visualization (Degrees of first 500 time steps of City 1)

| Vanilla LSTM  | LSTM with BiLinear Pooling | AutoencoderLSTM with BiLinear Pooling |
|---------------| -------------------------- | ------------------------------------- |
| ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/lstm_dir.png) | ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/lstm_bi_dir.png) | ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/ae_bi_dir.png) |


### References: Dataset source and project structure
- Dataset from "Trebing, Kevin and Mehrkanoon, Siamak, 2020, Wind speed prediction using multidimensional convolutional neural networks" [Github](https://github.com/HansBambel/multidim_conv)
- Project structure: [Github](https://github.com/ossez-com/python-project-structure-sample)

