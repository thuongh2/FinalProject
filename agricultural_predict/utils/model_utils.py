import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error as mae 

def save_model(model, model_url):
    joblib.dump(model, model_url)
    
    
def load_model(model_url):
    if model_url.endswith('.joblib'):
        return joblib.load(model_url)
    else:
        raise Exception("File không đúng định dạng")


def forecast_accuracy(forecast, actual):
    mae_score = mae(actual, forecast)
    mape = mae(actual, forecast) * 100  # MAPE     # ME
    rmse = np.mean((forecast - actual)**2)**.5  # RMS
    return({'mape':mape,'mae': mae_score, 'rmse':rmse})


