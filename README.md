# NL-weather

Here we see how temperature can have effect on wind speed and direction. we build Fusion neural network where temperature and wind time series data are used together in forcasting of wind speed and wind direction. 

### Data

- The dataset contains wind speed, wind direction, temperature, wind pressure, rain amount and Dew Point (6 features). 
- The data was acquired from 7 cities in Netherlands from 2011 to 2020 with 81.000 datapoints. 
- The data is split into training (January 2011 - December 2018) and testing (January 2019 - March 2020)

### Methods

- LSTM using only wind speed data

- LSTM + Bilinear Pooling using wind speed data + temperature data


### Results
[Check Demo](https://github.com/mhmdrdwn/NLweather/blob/main/demo.ipynb)

### Wind Speed

#### Error matrics

| Error | Model                      | 1H ahead | 5H ahead|10H ahead|50H ahead|
|-------| -------------------------- |:--------:|:-------:|:-------:|:-------:|
| MAE   | LSTM Baseline              |  8.99    |  13.73  |   17.74 |  18.91  |
| MAE   | LSTM+BiLinPooling          |  7.30    |  9.38   |   11.55 |  17.33  | 
| RMSE  | LSTM Baseline              | 11.84    |  17.77  |  22.18  |  24.19  |
| RMSE  | LSTM+BiLinPooling          | 10.82    |  13.43  |   16.18 |  22.81  |

#### sample visualization

| forcasting of wind speed using vanilla LSTM  | forcasting of wind speed using LSTM with BiLinear Pooling |
|----------------------------------------------| --------------------------------------------------------- |
| ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/lstm_speed.png) | ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/lstm_bi_speed.png)     |

### Wind Direction

#### Error matrics

| Error | Model             | 1H ahead | 5H ahead|10H ahead|50H ahead|
|-------| ----------------- |:--------:|:-------:|:-------:|:-------:|
| MAE   | LSTM Baseline     |  29.08   |  43.29  |   54.65 |  69.39  | 
| MAE   | LSTM+BiLinPooling |  22.95   |  34.20  |   41.69 |  65.93  | 
| RMSE  | LSTM Baseline     |  65.07   |  78.30  |   86.88 |  91.49  |
| RMSE  | LSTM+BiLinPooling |  58.74   |  70.59  |   77.69 |  93.05  |

#### sample visualization

| forecasting of wind direction (0:360) using vanilla LSTM  | forecasting of wind direction (0:360) using LSTM with BiLinear Pooling |
|-----------------------------------------------------------------| ---------------------------------------------------------------------------- |
| ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/lstm_dir.png) | ![alt text](https://github.com/mhmdrdwn/NLweather/blob/main/plots/lstm_bi_dir.png)     |


### References: Dataset source and project structure
- Dataset from "Trebing, Kevin and Mehrkanoon, Siamak, 2020, Wind speed prediction using multidimensional convolutional neural networks" [Github](https://github.com/HansBambel/multidim_conv)
- Project structure: [Github](https://github.com/ossez-com/python-project-structure-sample)

