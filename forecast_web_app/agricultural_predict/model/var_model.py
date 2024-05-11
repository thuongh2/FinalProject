import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import logging
from statsmodels.tsa.api import VAR

class VARModel:
    
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
        
        self.list_feature = []
        
        
    def predict(self, n_periods, df_forecast=None):
        self.model = joblib.load(self.model_url)
        if self.model:
            lag_order = self.model.k_ar
            print(lag_order)
            forecast_input = df_forecast.values[-lag_order:]
            fc = self.model.forecast(y=forecast_input, steps=n_periods)
            return fc
        else:
            raise Exception("Không tìm thấy model")


    def __predict_self_train(self, n_periods, df_forecast=None):
        if self.model:
            lag_order = self.model.k_ar
            print(lag_order)
            forecast_input = df_forecast.values[-lag_order:]
            fc = self.model.forecast(y=forecast_input, steps=n_periods)
            return fc
        else:
            raise Exception("Không tìm thấy model")



    def train_for_upload_mode(self, n_periods, test_data):
        df_diff = self.difference_dataset()
        self.forecast_data = self.predict(n_periods, df_diff)
        if self.forecast_data.size == 0:
            raise Exception("Không tìm thấy model")

        self.forecast_data = pd.DataFrame(self.forecast_data, index=self.test_data.index[-n_periods:], columns=self.test_data.columns)

        print(self.forecast_data.info())
        self.forecast_data = self.invert_transformation(self.train_data, self.forecast_data, second_diff=True)
        self.forecast_data['price'] = self.forecast_data['price_forecast']
        self.accuracy = self.forecast_accuracy(self.forecast_data.price_forecast.values, test_data.price.values)
        return self.forecast_data, self.accuracy

    def __prepare_data_for_self_train(self, split_size=0.8):
        logging.info('Start prepare data ' + self.data_uri)
        self.prepare_data(split_size)

    def train_model(self, argument):
        """
        Train model in web
        params argument
            size: size split data
            P: VAR(P)
        return ARIMA MODEL

        """

        self.__prepare_data_for_self_train(argument.get('size', 0.8))
        df_differenced = self.difference_dataset()

        logging.info('Start train VAR MODEL')

        model_search = VAR(df_differenced)

        aic_core = {}
        for i in range(1, argument['max_p'] + 1):
            result = model_search.fit(i)
            print('Lag Order =', i)
            print('AIC : ', result.aic)
            print('BIC : ', result.bic)
            print('FPE : ', result.fpe)
            print('HQIC: ', result.hqic, '\n')
            aic_core[i] = result.aic
        P = min(aic_core, key=aic_core.get)
        print('P : ', P)

        self.model = model_search.fit(P)

        n_periods = len(self.test_data)
        self.forecast_data = self.__predict_self_train(n_periods, df_differenced)

        self.forecast_data = pd.DataFrame(self.forecast_data, index=self.test_data.index[-n_periods:],
                                          columns=self.test_data.columns)

        print(self.forecast_data.info())
        self.forecast_data = self.invert_transformation(self.train_data, self.forecast_data, second_diff=True)
        self.forecast_data['price'] = self.forecast_data['price_forecast']
        self.accuracy = self.forecast_accuracy(self.forecast_data.price_forecast.values, self.test_data.price.values)
        return self.forecast_data, self.accuracy, self.model

    def __adjust(val, length=6):
        return str(val).ljust(length)

    def forecast_accuracy(self, test_data, predicted_values):
        mape = np.mean(np.abs((test_data - predicted_values) / test_data)) * 100
        mse = mean_squared_error(test_data, predicted_values)
        rmse = np.sqrt(mse)

        return {'mape': round(mape, 2), 'rmse': round(rmse, 2)}
    
    
    def difference_dataset(self, interval=None):
        df_dif = self.train_data.copy()
        return df_dif.diff().dropna()
    

    def prepare_data(self, train_url, test_url):
        if train_url:
            self.train_data = pd.read_csv(train_url)
        if test_url:
            self.test_data = pd.read_csv(test_url)

        return self.set_index_date(self.train_data, self.test_data)
    

    def prepare_data(self, split_size=0.8):
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



    def seasonal_diff(self, dataset, interval= 1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return diff


    def invert_transformation(self, df_train, df_forecast, second_diff=False):
        """Revert back the differencing to get the forecast to original scale."""
        df_fc = df_forecast.copy()
        columns = df_train.columns
        for col in columns:
            # Roll back 2nd Diff
            if second_diff:
                df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)].cumsum()
            # Roll back 1st Diff
            df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
        return df_fc


    def ml_flow_register(self):
        ARTIFACT_PATH = "model"

        mlflow.set_tracking_uri(uri="http://20.2.210.176:5000/")
        mlflow.set_experiment("VAR_MODEL")

        # Create an instance of a PandasDataset
        dataset = mlflow.data.from_pandas(
            self.data, source=self.data_uri, name="rice data", targets="price"
        )

        with mlflow.start_run() as run:
            mlflow.autolog(log_models=True)

            input_sample = pd.DataFrame(self.train_data)
            output_sample = pd.DataFrame(self.forecast_data)
            
            mlflow.log_input(dataset, context="training")
            
            mlflow.log_params({"P": self.model.k_ar})
            
            for k, v in self.accuracy.items():
                mlflow.log_metric(k, round(v,4))
            
            signature = infer_signature(input_sample, output_sample)

            model_info = mlflow.statsmodels.log_model(statsmodels_model=self.model,
                                                      signature=signature,
                                                      artifact_path="varmodel",
                                                      registered_model_name="statsmodels_model")
            return model_info


    
if __name__ == '__main__':
    data_url = "../test_data/var_data.csv"
    model_url = "../test_data/var_model.joblib"
    model = VARModel()
    model.model_url = model_url    
    model.data_uri = data_url
    # tạo data
    # _ , test_data = model.prepare_data(data_url)
    # xử lí dữ liệu (cho trai trên web)
    # print(test_data.head())
    # print(test_data.iloc[0].values)
    # dự đoná mô hình
    # data , ac = model.train_for_upload_mode(len(test_data), test_data)
    # register in ml flow
    # model_mflow = model.ml_flow_register()
    # print(model_mflow.model_uri)
    # print(ac)
    # print(data['price_forecast'])

    _, a, _ = model.train_model({'max_p': 10})
    print(a)

    