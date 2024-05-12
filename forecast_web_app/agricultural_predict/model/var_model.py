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
from base_model import BaseModel


class VARModel(BaseModel):
    
    def __init__(self):
        super().__init__()
        
        
    def predict(self, n_periods, df_forecast=None):
        if self.model:
            lag_order = self.model.k_ar
            forecast_input = df_forecast.values[-lag_order:]
            forecast = self.model.forecast(y=forecast_input, steps=n_periods)
            return forecast
        else:
            raise Exception("Không tìm thấy model")


    def forecast_future(self, forecast_num, data=None, n_steps=None):
        self._load_model()
        if self.model is None:
            raise Exception("Không tìm thấy model")
        
        n_periods = len(self.test_data) + forecast_num
        df_diff = self.difference_dataset()

        predict_data = self.predict(n_periods, df_diff)
        self.forecast_data = pd.DataFrame(predict_data, columns=self.test_data.columns)

        predicted = self.invert_transformation(self.train_data,  self.forecast_data, second_diff=True)
        predicted['price'] = predicted['price_forecast']

        last_date = self.test_data.index[-1]
        next_dates = pd.date_range(start=last_date, periods=forecast_num + 1)[1:]
        predicted = predicted[-forecast_num:]

        predicted_df = pd.DataFrame({'date': next_dates, 'price': predicted[self.PRICE_COLUMN]})
        
        return predicted_df

    def _load_model(self):
        self.model = joblib.load(self.model_url)


    def train_for_upload_mode(self, n_periods, data=None):
        self._load_model()
        if self.model is None:
            raise Exception("Không tìm thấy model")
        df_diff = self.difference_dataset()
        self.forecast_data = self.predict(n_periods, df_diff)
        if self.forecast_data.size == 0:
            raise Exception("Không tìm thấy model")

        self.forecast_data = pd.DataFrame(self.forecast_data, index=self.test_data.index[-n_periods:], columns=self.test_data.columns)

        self.forecast_data = self.invert_transformation(self.train_data, self.forecast_data, second_diff=True)
        self.forecast_data['price'] = self.forecast_data['price_forecast']
        self.accuracy = self.forecast_accuracy(self.forecast_data[self.PRICE_COLUMN].values, self.test_data[self.PRICE_COLUMN].values)

        return self.forecast_data, self.accuracy

  
    def train_model(self, argument):
        """
        Train model in web
        params argument
            size: size split data
            P: VAR(P)
        return ARIMA MODEL

        """

        self.prepare_data_for_self_train(argument.get('size', 0.8))
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
        self.forecast_data = self.predict(n_periods, df_differenced)

        self.forecast_data = pd.DataFrame(self.forecast_data, index=self.test_data.index[-n_periods:],
                                          columns=self.test_data.columns)

        print(self.forecast_data.info())
        self.forecast_data = self.invert_transformation(self.train_data, self.forecast_data, second_diff=True)
        self.forecast_data['price'] = self.forecast_data['price_forecast']
        self.accuracy = self.forecast_accuracy(self.forecast_data.price_forecast.values, self.test_data.price.values)
        return self.forecast_data, self.accuracy, self.model

    def __adjust(val, length=6):
        return str(val).ljust(length)

    
    def difference_dataset(self, interval=None):
        df_dif = self.train_data.copy()
        return df_dif.diff().dropna()


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
    model.prepare_data_for_self_train()

    # data , ac = model.train_for_upload_mode(168)

    model.forecast_future(10)
    # register in ml flow
    # model_mflow = model.ml_flow_register()
    # print(model_mflow.model_uri)
    # print(ac)
    # print(data['price_forecast'])

    _, a, _ = model.train_model({'max_p': 10})
    print(a)

    