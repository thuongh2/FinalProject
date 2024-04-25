import pandas as pd
from arima_model import ARIMAModel

if __name__ == '__main__':
    test_data_url = "../test_data/test_data_arima.csv"
    test_data = pd.read_csv("../test_data/test_data_arima.csv")
    model_url = "../test_data/arima.joblib"
    model = ARIMAModel()
    model.test_data = "../test_data/test_data_arima.csv"
    model.model_url = model_url    
    _ , test_data = model.prepare_data(None, test_data_url)
    data , ac = model.train_for_upload_mode(599, test_data)
    print(ac)
    print(data.head())
