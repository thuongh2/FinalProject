import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pmdarima as pm
import logging


class ARIMAModel:

    def __init__(self):
        # model
        self.model = None
        # link model
        self.model_url = ""
        # link dữ liệu
        self.data_uri = ""
        # dữ liệu gốc
        self.data = pd.DataFrame()
        # train data
        self.train_data = pd.DataFrame()
        # test data
        self.test_data = pd.DataFrame()
        # dự báo
        self.forecast_data = pd.DataFrame()
        # accuracy
        self.accuracy = {}

    def predict(self, n_periods=30):
        self.model = joblib.load(self.model_url)
        if self.model:
            predict, confint = self.model.predict(n_periods=n_periods, return_conf_int=True, dynamic=True)
            predict_series = pd.Series(predict, index=self.test_data.index)
            return predict_series
        else:
            raise Exception("Không tìm thấy model")

    def __predict_self_train(self, n_periods=30):
        if self.model:
            predict, confint = self.model.predict(n_periods=n_periods, return_conf_int=True, dynamic=True)
            predict_result = pd.DataFrame(predict, index=self.test_data.index, columns=['price'])
            return predict_result
        else:
            raise Exception("Không tìm thấy model")

    def train_for_upload_mode(self, n_periods, test_data):
        self.forecast_data = self.predict(n_periods)
        if self.forecast_data.empty:
            raise Exception("Không tìm thấy model")
        print(test_data.iloc[1])
        print(self.forecast_data.info())

        self.accuracy = self.forecast_accuracy(self.forecast_data, test_data.price.values)
        return self.forecast_data, self.accuracy

    def __prepare_data_for_self_train(self, split_size=0.8):
        logging.info('Start prepare data ' + self.data_uri)
        self.prepare_data(self.data_uri, split_size)

    def train_model(self, argument):
        """ 
        Train model in web
        params argument 
            size: size split data
            start_p: start p
            start_q: start q
            max_p: max p
            max_q: max q
        return ARIMA MODEL
        
        """

        if argument.get('size', 0.8) is None:
            raise Exception("Size is required")

        self.__prepare_data_for_self_train(argument['size'])

        logging.info('Start train ARIMA MODEL')

        self.model = pm.auto_arima(self.train_data.values, start_p=argument.get('start_p', 0),
                                   start_q=argument.get('start_q', 0),
                                   test='adf',  # use adftest to find optimal 'd'
                                   max_p=argument.get('max_p', 0), max_q=argument.get('max_q', 0),  # maximum p and q
                                   m=1,  # frequency of series
                                   d=argument.get('d', None),  # let model determine 'd'
                                   seasonal=False,
                                   start_P=0,
                                   D=0,
                                   trace=True,
                                   error_action='ignore',
                                   suppress_warnings=True,
                                   stepwise=True)

        n_periods = len(self.test_data)
        self.forecast_data = self.__predict_self_train(n_periods)

        if self.forecast_data.empty:
            raise Exception("Data predict not found")

        print(self.forecast_data.info())

        self.accuracy = self.forecast_accuracy(self.forecast_data.price.values, self.test_data.price.values)
        return self.forecast_data, self.accuracy, self.model

    def forecast_accuracy(self, test_data, predicted_values):
        mape = np.mean(np.abs((test_data - predicted_values) / test_data)) * 100
        mse = mean_squared_error(test_data, predicted_values)
        rmse = np.sqrt(mse)

        return {'mape': round(mape, 2), 'rmse': round(rmse, 2)}

    # def prepare_data(self, train_url, test_url):
    #     if train_url:
    #         self.train_data = pd.read_csv(train_url)
    #     if test_url:
    #         self.test_data = pd.read_csv(test_url)
    #
    #     return self.set_index_date(self.train_data, self.test_data)

    def prepare_data(self, data_url, split_size=0.8):
        logging.info('Read data from ' + data_url)
        self.data = pd.read_csv(data_url)

        if self.data.empty:
            raise Exception(f'Data {data_url} empty')

        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)

        size = int(len(self.data) * split_size)

        self.train_data = self.data[:size]
        self.test_data = self.data[size:]

        return self.train_data, self.test_data

    def set_index_date(self, train_data, test_data):
        if 'date' in train_data.columns:
            train_data['date'] = pd.to_datetime(train_data['date'])
            train_data.set_index('date', inplace=True)

        if 'date' in test_data.columns:
            test_data['date'] = pd.to_datetime(test_data['date'])
            test_data.set_index('date', inplace=True)

        return train_data, test_data

    def seasonal_diff(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return diff

    # invert differenced forecast
    def inverse_difference(self, last_ob, value):
        return value + last_ob

    def ml_flow_register(self, experient_name="DEFAUT_MODEL"):
        ARTIFACT_PATH = "model"

        mlflow.set_tracking_uri(uri="http://20.2.210.176:5000/")
        mlflow.set_experiment(experient_name)

        # Create an instance of a PandasDataset
        dataset = mlflow.data.from_pandas(
            self.data, source=self.data_uri, name="rice data", targets="price"
        )

        with mlflow.start_run() as run:
            input_sample = pd.DataFrame(self.train_data)
            output_sample = pd.DataFrame(self.forecast_data)

            mlflow.log_input(dataset, context="training")

            mlflow.log_params({"order": self.model.order})
            mlflow.log_params({"model": self.model.summary()})

            for k, v in self.accuracy.items():
                mlflow.log_metric(k, round(v, 4))

            signature = infer_signature(input_sample, output_sample)

            model_mflow = mlflow.pmdarima.log_model(
                model, ARTIFACT_PATH, signature=signature
            )
            return model_mflow


if __name__ == '__main__':
    # test_data_url = "../test_data/test_data_arima.csv"
    # test_data = pd.read_csv("../test_data/test_data_arima.csv")
    data_url = "../test_data/arima_data.csv"
    model_url = "../test_data/arima.joblib"
    model = ARIMAModel()
    # model.test_data = "../test_data/test_data_arima.csv"
    # model.model_url = model_url
    # model.data_uri = data_url
    # # tạo data
    # _ , test_data = model.prepare_data(data_url)
    # # xử lí dữ liệu (cho trai trên web)
    # print(test_data.head())
    # print(test_data.iloc[0].values)
    # # dự đoná mô hình
    # data , ac = model.train_for_upload_mode(599, test_data)
    # # register in ml flow
    # model_mflow = model.ml_flow_register()
    # print(model_mflow.model_uri)
    # print(ac)
    # print(data)
    model.data_uri = data_url
    model.train_model({'start_p': 1, 'start_q': 1, 'max_p': 1, 'max_q': 1, 'size': 0.8})
    model.ml_flow_register()
