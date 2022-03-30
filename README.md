# NL-weather
Here we use how temperature can have effect on wind speed and direction. we build Fusion neural network where temperature and wind time series data are used togethr in forcasting of wind speed and wind direction. 

### Data

The dataset contains wind speed, wind direction, temperature, wind pressure  (6 features). 

- The data was acquired from 7 cities in Netherlands

### Methods

- LSTM using only wind speed data

- LSTM + Bilinear Pooling using wind speed data + temperature data


### Results
### Wind Speed

[Check Demo](https://github.com/mhmdrdwn/NLweather/blob/main/wind_speed_demo.ipynb)

| Model vs Error |		MAE	       	                      |     RMSE                               |
| -------------- |:--------------------------------------:|:--------------------------------------:|
| Model          | 1H ahead | 5H ahead|10H ahead|50H ahead| 1H ahead |5H ahead |10H ahead|50H ahead|
| -------------- |:--------:|:-------:|:-------:|:-------:|:--------:|:-------:|:-------:|:-------:|
| LSTM Baseline  |  8.99    |  13.73  |   17.74 |  18.91  |  11.84   |  17.77  |  22.18  |  24.19  |
| LSTM+BiPooling |  7.30    |  9.38   |   11.55 |  17.33  |  10.82   |  13.43  |   16.18 |  22.81  |



### Wind Direction


| Model         | 1 Hour ahead | 5 Hours ahead|10 Hours ahead|50 Hours ahead|
| ------------- |:------------:|:------------:|:------------:|:------------:|
| LSTM          |  29.57       |  43.29       |   54.87      |  67.06       |
| LSTM+BiPooling|  10.86       |  13.43       |   16.18      |  22.81       |



### References: Dataset source and project structure
- Dataset from "Trebing, Kevin and Mehrkanoon, Siamak, 2020, Wind speed prediction using multidimensional convolutional neural networks" [Github](https://github.com/HansBambel/multidim_conv)
- Project structure: [Github](https://github.com/ossez-com/python-project-structure-sample)

