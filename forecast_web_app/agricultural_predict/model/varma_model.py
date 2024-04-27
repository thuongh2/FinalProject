import joblib
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class VARMAModel:

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
            fc = self.model.forecast(steps=n_periods)
            return fc
        else:
            raise Exception("Không tìm thấy model")

    def train_for_upload_mode(self, n_periods, test_data):
        self.difference_dataset()
        self.forecast_data = self.predict(n_periods)
        if self.forecast_data.size == 0:
            raise Exception("Không tìm thấy model")

        print(self.forecast_data.info())
        self.forecast_data = self.invert_transformation(self.train_data[self.list_feature], self.forecast_data,
                                                        second_diff=True)

        self.accuracy = self.forecast_accuracy(self.forecast_data.price.values, test_data.price.values)
        return self.forecast_data, self.accuracy

    def forecast_accuracy(self, forecast, actual):
        mse = mean_squared_error(actual, forecast)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, forecast)
        return ({'mse': round(mse, 3), 'rmse': round(rmse, 3), 'r2': round(r2, 3)})

    def difference_dataset(self, interval=None):
        self.train_data = self.train_data.diff().dropna()

    def prepare_data(self, train_url, test_url):
        if train_url:
            self.train_data = pd.read_csv(train_url)
        if test_url:
            self.test_data = pd.read_csv(test_url)

        return self.set_index_date(self.train_data, self.test_data)

    def prepare_data(self, data_url):
        self.data = pd.read_csv(data_url)
        print(self.data.info())
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)
        size = int(len(self.data) * 0.8)
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

    def invert_transformation(self, df_train, df_forecast, second_diff=False):
        """Revert back the differencing to get the forecast to original scale."""
        df_fc = df_forecast.copy()
        columns = df_train.columns
        for col in columns:
            # Roll back 2nd Diff
            if second_diff:
                df_fc[str(col) + '_1d'] = (df_train[col].iloc[-1] - df_train[col].iloc[-2]) + df_fc[str(col)].cumsum()
            # Roll back 1st Diff
            df_fc[str(col) + '_forecast'] = df_train[col].iloc[-1] + df_fc[str(col) + '_1d'].cumsum()
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

            # mlflow.log_params({"P": self.model.k_ar})

            for k, v in self.accuracy.items():
                mlflow.log_metric(k, round(v, 4))

            signature = infer_signature(input_sample, output_sample)

            model_info = mlflow.statsmodels.log_model(statsmodels_model=self.model,
                                                      signature=signature,
                                                      artifact_path="varmodel",
                                                      registered_model_name="statsmodels_model")
            return model_info


if __name__ == '__main__':
    data_url = "../test_data/var_data.csv"
    model_url = "../test_data/varma_model.joblib"
    model = VARMAModel()
    model.model_url = model_url
    model.data_uri = data_url
    # tạo data
    _, test_data = model.prepare_data(data_url)
    # xử lí dữ liệu (cho trai trên web)
    print(test_data.head())
    print(test_data.iloc[0].values)
    # dự đoná mô hình
    data, ac = model.train_for_upload_mode(len(test_data), test_data)
    # register in ml flow
    model_mflow = model.ml_flow_register()
    print(model_mflow.model_uri)
    print(ac)
    print(data)

