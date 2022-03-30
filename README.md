# NL-weather
Here i'm learning weather forcasting using Deep learning methods

## Wind Speed

### Methods

- LSTM using only wind speed data

- LSTM + Bilinear Pooling using wind speed data + temperature data


### Results

[Check Demo](https://github.com/mhmdrdwn/NLweather/blob/main/wind_speed_demo.ipynb)

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




- Dataset from "Trebing, Kevin and Mehrkanoon, Siamak, 2020, Wind speed prediction using multidimensional convolutional neural networks" [Github](https://github.com/HansBambel/multidim_conv)
- Project structure: [Github](https://github.com/ossez-com/python-project-structure-sample)

