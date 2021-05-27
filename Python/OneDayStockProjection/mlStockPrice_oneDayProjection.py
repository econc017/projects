import pandas
import pandas_datareader as webData
import numpy
import matplotlib.pyplot as plot
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python.ops.array_ops import tensor_scatter_nd_update


def loadData(company: str, start: dt.datetime, end: dt.datetime):
    company = company
    data = webData.DataReader(company, 'yahoo', start, end)
    return data


def prepareData(data, company: str):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledData = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    predictionDays = 60

    xTrain = []
    yTrain = []

    for x in range(predictionDays, len(scaledData)):
        xTrain.append(scaledData[x-predictionDays:x, 0])
        yTrain.append(scaledData[x, 0])

    xTrain, yTrain = numpy.array(xTrain), numpy.array(yTrain)
    xTrain = numpy.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

    # Build Model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True,
              input_shape=(xTrain.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of next closing price

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(xTrain, yTrain, epochs=25, batch_size=32)

    ''' Test Accuracy on Existing Data '''
    # Load test data
    testStart = dt.datetime(2020, 5, 1)
    testEnd = dt.datetime.now()

    testData = webData.DataReader(company, 'yahoo', testStart, testEnd)
    actualPrices = testData['Close'].values

    totalDataSet = pandas.concat((data['Close'], testData['Close']))

    modelInputs = totalDataSet[len(
        totalDataSet) - len(testData) - predictionDays:].values
    modelInputs = modelInputs.reshape(-1, 1)
    modelInputs = scaler.transform(modelInputs)

    # Make predictions on test data
    xTest = []

    for x in range(predictionDays, len(modelInputs)):
        xTest.append(modelInputs[x-predictionDays:x, 0])

    xTest = numpy.array(xTest)
    xTest = numpy.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

    predictedPrices = model.predict(xTest)
    predictedPrices = scaler.inverse_transform(predictedPrices)

    # Plot the test predictions
    plot.plot(actualPrices, color="black", label="actual")
    plot.plot(predictedPrices, color="green", label="prediction")
    plot.title(f"{company} Share price")
    plot.xlabel("Time")
    plot.ylabel(f"{company} Share Price")
    plot.legend()
    plot.show()


if __name__ == "__main__":
    startTrainingDate = dt.datetime(2018, 1, 1)
    endTrainingDate = dt.datetime(2020, 5, 1)
    data = loadData("AAPL", startTrainingDate, endTrainingDate)
    preparedData = prepareData(data, "AAPL")
