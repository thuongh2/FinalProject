import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc



def predict(model_url, n_periods, train_data, test_data):
    model_load = joblib.load(model_url)
    if model_load:
        lag_order = model_load.k_ar
        print(lag_order)
        # Input data for forecasting
        forecast_input = convert_to_stationary(train_data).values[-lag_order:]
        print(forecast_input)
        
        fc = model_load.forecast(y=forecast_input, steps=n_periods)
        df_forecast = pd.DataFrame(fc, index=test_data.index[-n_periods:], columns=test_data.columns + '_2d')
        df_results = invert_transformation(train_data, df_forecast, second_diff=True)
        df_results.loc[:, ['price_forecast']]
        return df_results
    else:
        raise Exception("Không tìm thấy model")


def convert_to_stationary(df_train, lags=1):
    df_differenced = df_train.diff().dropna()
    return df_differenced

def train_model():
    pass


def evaluate_mode():
    pass


from sklearn.metrics import mean_absolute_error as mae 

def forecast_accuracy(forecast, actual):
    mae = mae(actual, forecast)
    mape =mae * 100  # MAPE     # ME
    rmse = np.mean((forecast - actual)**2)**.5  # RMS
    return({'mape':mape,'mae': mae, 'rmse':rmse})
    
    
def train_for_upload_mode(model_url, n_periods, train_data, test_data):
    
    fc = predict(model_url, n_periods, train_data, test_data)
    print(fc.price_forecast.values)
    score = forecast_accuracy(fc.price_forecast.values, test_data.price[:n_periods].values)
    print(score)
    return fc, score
    
      
if __name__ == '__main__':
    test_data = pd.read_csv("../test_data/test_var_model.csv")
    train_data = pd.read_csv("../test_data/train_var_model.csv")
    train_data['date'] = pd.to_datetime(train_data['date'])
    train_data = train_data.set_index('date')
    test_data['date'] = pd.to_datetime(test_data['date'])
    test_data = test_data.set_index('date')
    model_url = "../test_data/var.joblib"
    train_for_upload_mode(model_url, 30,train_data, test_data)