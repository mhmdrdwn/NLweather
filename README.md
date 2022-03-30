# NL-weather
Here i'm learning weather forcasting using Deep learning methods

## Wind Speed

### Methods

[LSTM using only wind speed data](https://github.com/mhmdrdwn/NL-weather/blob/main/wind_speed/lstm.ipynb)

[LSTM + Bilinear Pooling using wind speed data + temperature data](https://github.com/mhmdrdwn/NL-weather/blob/main/wind_speed/poolinglstm.ipynb)


### Results

#### MAE
| Model          | 1 Hour ahead | 5 Hours ahead|10 Hours ahead|50 Hours ahead|
| -------------- |:------------:|:------------:|:------------:|:------------:|
| LSTM           |  9.15        |  13.72       |   17.43      |  19.09       |
| LSTM+BiPooling |  7.10        |  9.18        |   11.49      |	 17.54       |

## Wind Direction

### Results

#### MAE

| Model         | 1 Hour ahead | 5 Hours ahead|10 Hours ahead|50 Hours ahead|
| ------------- |:------------:|:------------:|:------------:|:------------:|
| LSTM          |  29.57       |  43.29       |   54.87      |  67.06       |


Project structure: from https://github.com/ossez-com/python-project-structure-sample

