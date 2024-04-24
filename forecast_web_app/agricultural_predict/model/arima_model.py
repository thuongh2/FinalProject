import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from sklearn.metrics import r2_score


def predict(model_url, n_periods, test_data):
    model_load = joblib.load(model_url)
    if model_load:
        fc, confint = model_load.predict(n_periods=n_periods, return_conf_int=True, dynamic=True)
        index_of_fc = np.arange(len(test_data), len(test_data.values) + n_periods)
        fc_series = pd.Series(fc, index=test_data.index)
        return fc_series
    else:
        raise Exception("Không tìm thấy model")


def train_model():
    pass


def evaluate_mode():
    pass



def train_for_upload_mode(model_url, n_periods, test_data):
    fc = predict(model_url, n_periods, test_data)
    score = forecast_accuracy(fc.values, test_data.price.values)
    print(score)
    return fc, score
    

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    r2 = r2_score(actual, forecast) 
    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse,  'r2': r2})
    
    
if __name__ == '__main__':
    test_data = pd.read_csv("../test_data/test_data_arima.csv")
    model_url = "../test_data/arima.joblib"
    train_for_upload_mode(model_url, 599, test_data)