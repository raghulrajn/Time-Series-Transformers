import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, accuracy_score

def sliding_window(arr, window_size = 1, step=0):
    """Assuming a time series with time advancing along dimension 0,
	window the time series with given size and step.

    :param arr : input array.
    :type arr: numpy.ndarray
    :param window_size: size of sliding window.
    :type window_size: int
    :param step: step size of sliding window. If 0, step size is set to obtain 
        non-overlapping contiguous windows (that is, step=window_size). 
        Defaults to 0.
    :type step: int

    :return: array 
    :rtype: numpy.ndarray
    """
    n_obs = arr.shape[0]

    # validate arguments
    if window_size > n_obs:
        raise ValueError(
            "Window size must be less than or equal "
            "the size of array in first dimension."
        )
    if step < 0:
        raise ValueError("Step must be positive.")

    n_windows = 1 + int(np.floor( (n_obs - window_size) / step))

    obs_stride = arr.strides[0]
    windowed_row_stride = obs_stride * step

    new_shape = (n_windows, window_size) + arr.shape[1:]
    new_strides = (windowed_row_stride, ) + arr.strides

    strided = np.lib.stride_tricks.as_strided(
        arr,
        shape=new_shape,
        strides=new_strides,
    )
    return strided

def selecting_req_signals(df, reqSignals):
    df = df[reqSignals]
    return df

def scale_data(trainDataFrame, testDataFrame,req_input_signal, req_output_signal):
    """scales the features of train and test data within -1 to 1. 
    for example, vehicle velocity ranges 0 to 100, while engine rpm varies from 700 to 3000. 

    so converting all the data within -1 to 1 for the purpose of training

    :param trainDataFrame : dataframe of mf4 files in train data path
    :type trainDataFrame: pd.DataFrame
    :param testDataFrame : dataframe of mf4 files in test data path
    :type testDataFrame: pd.DataFrame
    :param req_input_signal : list of input features
    :type req_input_signal: list
    param req_output_signal : list of output features
    :type req_output_signal: list
    
    :return: traindataframe, testdataframe, inputscaler, outputscaler
    :rtype: pd.Dataframe, pd.Dataframe, sklearn.MinMaxScaler, sklearn.MinMaxScaler
    """

    input_scaler = MinMaxScaler(feature_range=(-1,1))
    output_scaler = MinMaxScaler(feature_range=(-1,1))
    
    reqSignals = req_input_signal + req_output_signal
    
    transformedTrainData = pd.DataFrame(columns=reqSignals)
    transformedTestData = pd.DataFrame(columns=reqSignals)

    transformedTrainData[req_input_signal] = input_scaler.fit_transform(trainDataFrame[req_input_signal])
    transformedTrainData[req_output_signal] = output_scaler.fit_transform(trainDataFrame[req_output_signal])

    transformedTestData[req_input_signal] = input_scaler.transform(testDataFrame[req_input_signal])
    transformedTestData[req_output_signal] = output_scaler.transform(testDataFrame[req_output_signal])

    return transformedTrainData, transformedTestData, input_scaler, output_scaler

def read_csv_file(dataPath, reqSignals):
    """reads MF4 files from a particular path and converts into dataframe

    :param datapath : input path of mf4 files.
    :type arr: str
    :param reqSignals: list of signlas required 
    :type reqSignals: list
    
    :return: concatenated dataframe of all csv files in the given path
    :rtype: pd.Dataframe
    """
    dataFrame = pd.DataFrame(columns=reqSignals)
    for file in os.listdir(dataPath):
        path = os.path.join(dataPath, file)
        signalDf = pd.read_csv(path)
        signalDf = signalDf[reqSignals]
        dataFrame = pd.concat([dataFrame,signalDf])
    return dataFrame


def compute_metrics(output, target):
    """computes mean square error, mean absolute error, mean apsolute percentage error and accuracy

    :param output : array of output predicted by the transformer model
    :type output: np.ndarray
    :param target : array of output expected
    :type target: np.ndarray

    :return: mse, mae, mape, accuracy
    :rtype: flat, float, float, float
    """
    output = output.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    # Reshape to 2D arrays
    output = output.reshape(-1, output.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    mse = mean_squared_error(target, output)
    mae = mean_absolute_error(target, output)
    mape = mean_absolute_percentage_error(target, output)
    # Placeholder for accuracy - depends on specific problem
    accuracy = np.mean((np.abs(target - output) < 0.1).all(axis=1))  # Adjust threshold and axis based on problem
    return mse, mae, mape, accuracy

