import joblib
import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pmdarima as pm
import logging
from model.base_model import BaseModel

class ARIMAXModel(BaseModel):

    def __init__(self):
        super().__init__()

        self.exogenous = []


    def predict(self, n_periods=30):
        if self.model:
            predict, confint = self.model.predict(n_periods=n_periods, return_conf_int=True, dynamic=True)
            return predict, confint
        else:
            raise Exception("Không tìm thấy model")


    def forecast_future(self, forecast_num, data=None, n_steps=None):
        self._load_model()
        if self.model is None:
            raise Exception("Không tìm thấy model")
        
        n_periods = len(self.test_data) + forecast_num
        predict_data, _ = self.predict(n_periods)

        last_date = self.test_data.index[-1]
        next_dates = pd.date_range(start=last_date, periods=forecast_num + 1)[1:]
        predicted = predict_data[-forecast_num:]

        predicted_df = pd.DataFrame({'date': next_dates, 'price': predicted})

        return predicted_df

    def load_model(self):
        self.model = joblib.load(self.model_url)

    def _load_model(self):
        self.model = joblib.load(self.model_url)


    def train_for_upload_mode(self, n_periods, data=None):
        self._load_model()
        if self.model is None:
            raise Exception("Không tìm thấy model")
        
        predict_data, _ = self.predict(n_periods)
        self.forecast_data = pd.DataFrame(predict_data, index=self.test_data.index, columns=[self.PRICE_COLUMN])
        if self.forecast_data.empty:
            raise Exception("Dự đoán lỗi")

        self.accuracy = self.forecast_accuracy(self.forecast_data.price.values, self.test_data.price.values)
        return self.forecast_data, self.accuracy


    def train_model(self, argument):
        """
        Train model in web
        params argument
            size: size split data
            start_p: start p
            start_q: start q
            max_p: max p
            max_q: max q
            set exogenous list
        return ARIMA MODEL

        """

        if argument.get('size', 0.8) is None:
            raise Exception("Size is required")

        self.__prepare_data_for_self_train(argument['size'])
        if argument.get('smoothing_data'):
            self.smoothing_data(argument.get('smoothing_data'), argument.get('smoothing_value'))

        logging.info('Start train ARIMA MODEL')

        self.model = pm.auto_arima(self.train_data.price.values,
                                   exogenous=self.train_data[self.exogenous],
                                   start_p=argument.get('start_p', 0),
                                   start_q=argument.get('start_q', 0),
                                   test='adf',  # use adftest to find optimal 'd'
                                   max_p=argument.get('max_p', 0),
                                   max_q=argument.get('max_q', 0),  # maximum p and q
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
        predict_data, _ = self.predict(n_periods)

        self.forecast_data = pd.DataFrame(predict_data, index=self.test_data.index, columns=[self.PRICE_COLUMN])
        if self.forecast_data is None:
            raise Exception("Data predict not found")

        self.accuracy = self.forecast_accuracy(self.forecast_data[self.PRICE_COLUMN].values, self.test_data[self.PRICE_COLUMN].values)
        return self.forecast_data, self.accuracy, self.model

    def difference_dataset(self, type, lag=1):
        """
            Dif data set
            LOG: Log Differencing
            DIFF: Difference Data
                LAG: number lag 
            AUTO: Automatic Differencing
        """
        df_diff = self.data.copy()
       
        if (type == 'LOG'):
            df_diff = np.log(df_diff)
        elif (type == 'AUTO'):
            df_diff = df_diff.diff()
        else:
            df_diff = df_diff - df_diff.shift(int(lag))
        df_diff = df_diff.dropna()
        return df_diff

    def seasonal_diff(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return diff

    # invert differenced forecast
    def inverse_difference(self, last_ob, value):
        return value + last_ob

    def ml_flow_param(self):
        return {"order": self.model.order}

    def ml_flow_register(self):
        ARTIFACT_PATH = "model"

        mlflow.set_tracking_uri(uri="http://agricultural.io.vn:5000/")
        mlflow.set_experiment("ARIMA_MODEL")

        # Create an instance of a PandasDataset
        dataset = mlflow.data.from_pandas(
            self.data, source=self.data_uri, name="rice data", targets="price"
        )

        with mlflow.start_run() as run:
            input_sample = pd.DataFrame(self.train_data)
            output_sample = pd.DataFrame(self.forecast_data)

            mlflow.log_input(dataset, context="training")

            mlflow.log_params({"order": self.model.order})

            for k, v in self.accuracy.items():
                mlflow.log_metric(k, round(v, 4))

            signature = infer_signature(input_sample, output_sample)

            model_mflow = mlflow.pmdarima.log_model(
                model, ARTIFACT_PATH, signature=signature
            )
            return model_mflow


if __name__ == '__main__':
    test_data_url = "../test_data/test_data_arima.csv"
    test_data = pd.read_csv("../test_data/test_data_arima.csv")
    data_url = "../test_data/arima_data.csv"
    model_url = "../test_data/arima.joblib"
    model = ARIMAModel()
    model.test_data = "../test_data/test_data_arima.csv"
    model.model_url = model_url
    model.data_uri = data_url
    # tạo data
    _, test_data = model.prepare_data(data_url)
    # xử lí dữ liệu (cho trai trên web)
    print(test_data.head())
    print(test_data.iloc[0].values)
    # dự đoná mô hình
    data, ac = model.train_for_upload_mode(599, test_data)
    # register in ml flow
    model_mflow = model.ml_flow_register()
    print(model_mflow.model_uri)
    print(ac)
    print(data)



