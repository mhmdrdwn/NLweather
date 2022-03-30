# NL-weather
Here we use how temperature can have effect on wind speed and direction. we build Fusion neural network where temperature and wind time series data are used togethr in forcasting of wind speed and wind direction. 

### Methods

- LSTM using only wind speed data

- LSTM + Bilinear Pooling using wind speed data + temperature data


### Results

[Check Demo](https://github.com/mhmdrdwn/NLweather/blob/main/wind_speed_demo.ipynb)

### Wind Speed
#### MAE

| Model          | 1 Hour ahead | 5 Hours ahead|10 Hours ahead|50 Hours ahead|
| -------------- |:------------:|:------------:|:------------:|:------------:|
| LSTM           |  8.98        |  13.73       |   17.74      |  18.91       |
| LSTM+BiPooling |  7.16        |  9.38        |   11.55      |	 17.33       |

#### RMSE

| Model          | 1 Hour ahead | 5 Hours ahead|10 Hours ahead|50 Hours ahead|
| -------------- |:------------:|:------------:|:------------:|:------------:|
| LSTM           |  11.82       |  17.77       |   22.18      |  24.19       |
| LSTM+BiPooling |  10.86       |  13.43       |   16.18      |  22.81       |



### Wind Direction

#### MAE

| Model         | 1 Hour ahead | 5 Hours ahead|10 Hours ahead|50 Hours ahead|
| ------------- |:------------:|:------------:|:------------:|:------------:|
| LSTM          |  29.57       |  43.29       |   54.87      |  67.06       |
| LSTM+BiPooling|  10.86       |  13.43       |   16.18      |  22.81       |


#### RMSE

| Model         | 1 Hour ahead | 5 Hours ahead|10 Hours ahead|50 Hours ahead|
| ------------- |:------------:|:------------:|:------------:|:------------:|
| LSTM          |  29.57       |  43.29       |   54.87      |  67.06       |
| LSTM+BiPooling|  10.86       |  13.43       |   16.18      |  22.81       |


### References: Dataset source and project structure
- Dataset from "Trebing, Kevin and Mehrkanoon, Siamak, 2020, Wind speed prediction using multidimensional convolutional neural networks" [Github](https://github.com/HansBambel/multidim_conv)
- Project structure: [Github](https://github.com/ossez-com/python-project-structure-sample)

