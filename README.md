# NL-weather
Here i'm learning weather forcasting using ML methods

## Wind Speed

### Methods

[LSTM using only wind speed data](https://github.com/mhmdrdwn/NL-weather/blob/main/wind_speed/lstm.ipynb)

[LSTM + Bilinear Pooling using wind speed data + temperature data](https://github.com/mhmdrdwn/NL-weather/blob/main/wind_speed/poolinglstm.ipynb)


### Results

#### MAE
| Model          | 1 Hour ahead | 5 Hours ahead|10 Hours ahead|50 Hours ahead|
| -------------- |:------------:|:------------:|:------------:|:------------:|
| LSTM           |  9.15        |  13.72       |   17.43      |  19.09       |
| LSTM+BilPooling|  7.49        |  9.37        |   11.65      |	 17.44       |

## Wind Direction

### Results

#### MAE

| Model         | 1 Hour ahead | 5 Hours ahead|10 Hours ahead|50 Hours ahead|
| ------------- |:------------:|:------------:|:------------:|:------------:|
| LSTM          |  29.57       |  43.29       |   54.87      |  67.06       |

