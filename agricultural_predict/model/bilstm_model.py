import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from model.base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import logging
import mlflow
from mlflow.models import infer_signature
import mlflow
from mlflow.models import infer_signature
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class BiLSTMModel(BaseModel):
    def __init__(self):
        super().__init__()
    
    def predict(self, data, n_steps):
        self.model = load_model(self.model_url)
        
        prices = data['price'].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices.reshape(-1, 1))
        X = []
        for i in range(n_steps, len(scaled_data)):
            X.append(scaled_data[i-n_steps:i, 0])
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        predicted_values = self.model.predict(X)
        predicted_values = scaler.inverse_transform(predicted_values)
        
        return predicted_values

    def load_model(self):
        self.model = load_model(self.model_url)

    def train_for_upload_mode(self, n_periods, test_data):
        input_shape = self.model.layers[0].input_shape
        self.time_steps = input_shape[1]
        forecast = self.predict(test_data, self.time_steps)
        forecast = np.concatenate([test_data.iloc[:self.time_steps]['price'].values, forecast.flatten()])
        self.forecast_data = pd.DataFrame(forecast, columns=['price'])
        self.forecast_data.set_index(test_data.index, inplace=True)

        if self.forecast_data.empty:
            raise Exception("Không tìm thấy model")
        print(self.forecast_data.info())

        self.accuracy = self.forecast_accuracy(self.test_data.price.values, self.forecast_data.price.values)
        return self.forecast_data, self.accuracy

    def forecast_future(self, forecast_num, data=None, time_steps=None):
        self.model = load_model(self.model_url)

        input_shape = self.model.layers[0].input_shape
        self.time_steps = input_shape[1]

        scaler_price = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler_price.fit_transform(self.data[self.PRICE_COLUMN].values.reshape(-1, 1))

        if len(self.data.columns) == 1 and self.PRICE_COLUMN in self.data.columns:
            last_data = scaled_data[-(self.time_steps + 1):]
            last_data = last_data.reshape(1, -1)[:, -((self.time_steps + 1) - 1):]
        else:
            other_columns = self.data.drop(columns=[self.PRICE_COLUMN])
            scalers_other = {}
            scaled_other = np.zeros(other_columns.shape)
            for i, col in enumerate(other_columns.columns):
                scalers_other[col] = MinMaxScaler(feature_range=(0, 1))
                scaled_other[:, i] = scalers_other[col].fit_transform(
                    other_columns[col].values.reshape(-1, 1)).flatten()
            scaled_data = np.concatenate((scaled_data, scaled_other), axis=1)
            last_data = scaled_data[-(self.time_steps + 1):]
            last_data = last_data.reshape(1, -1, scaled_data.shape[1])[:, -((self.time_steps + 1) - 1):]

        predicted_prices = []
        for day in range(forecast_num):
            next_prediction = self.model.predict(last_data)
            if len(self.data.columns) == 1 and self.PRICE_COLUMN in self.data.columns:
                last_data = np.append(last_data, next_prediction).reshape(1, -1)[:, 1:]
            else:
                last_features = last_data[0, -1, 1:]
                next_prediction_full = np.append(next_prediction, last_features).reshape(1, 1, -1)
                last_data = np.append(last_data, next_prediction_full, axis=1)
                last_data = last_data[:, 1:, :]
                
            predicted_price = scaler_price.inverse_transform(next_prediction.reshape(-1, 1))
            predicted_prices.append(predicted_price[0, 0])

        last_date = self.data.index[-1]
        next_dates = pd.date_range(start=last_date, periods=forecast_num + 1)[1:]
        predicted_df = pd.DataFrame({'date': next_dates, 'price': predicted_prices})

        return predicted_df

    def concat_dataframes(self, original_df, predicted_df):
        predicted_df['date'] = pd.to_datetime(predicted_df['date'])
        predicted_df['date'] = predicted_df['date'].dt.strftime('%m/%d/%Y')
        original_df['date'] = pd.to_datetime(original_df['date'])
        original_df['date'] = original_df['date'].dt.strftime('%m/%d/%Y')
        data_predicted = pd.concat([original_df, predicted_df], ignore_index=True)
        
        return data_predicted
    
    def create_sequences(self, data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length), :])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)

    def create_model(self, argument, input_shape):
        self.model = Sequential()
        layers_data = argument.get('layers_data', [{'id': 0, 'units': 64}])
        layers_data = [{'id': layer['id'], 'units': int(layer['units'])} for layer in layers_data]

        if len(layers_data) == 1:
            units = layers_data[0]['units']
            self.model.add(Bidirectional(LSTM(units, return_sequences=False), input_shape=input_shape))
            self.model.add(Dropout(0.2))
        else:
            for i, layer in enumerate(layers_data):
                units = layer['units']
                if i == 0:
                    self.model.add(Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape))
                    self.model.add(Dropout(0.2))
                elif i == len(layers_data) - 1:
                    self.model.add(Bidirectional(LSTM(units, return_sequences=False)))
                    self.model.add(Dropout(0.2))
                else:
                    self.model.add(Bidirectional(LSTM(units, return_sequences=True)))
                    self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        optimizer = Adam()
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def train_model(self, argument):
        """
        Train model in web
        params argument
            size: size split data
            layer: list of layer configurations
            epoch: number of epochs
            batch_size: size of the batch
        return BiLSTM MODEL
        """
        # Prepare data
        if argument.get('size', 0.8) is None:
            raise Exception("Size is required")
        
        self.data = pd.read_csv(self.data_uri)
        self._clean_data()
        
        if argument.get('smoothing_data'):
            self.smoothing_data(argument.get('smoothing_data'), argument.get('smoothing_value'))

        time_step = argument.get('timestep', 10)

        # Check data column
        if len(self.data.columns) == 1 and self.PRICE_COLUMN in self.data.columns:
            scaler_price = MinMaxScaler(feature_range=(0, 1))
            scaled_price = scaler_price.fit_transform(self.data[self.PRICE_COLUMN].values.reshape(-1, 1))

            self.X, self.y = self.create_sequences(scaled_price, time_step)
            input_shape = (time_step, 1)
        else:
            scaler_price = MinMaxScaler(feature_range=(0, 1))
            scaled_price = scaler_price.fit_transform(self.data[self.PRICE_COLUMN].values.reshape(-1, 1))

            other_columns = self.data.drop(columns=[self.PRICE_COLUMN])
            scalers_other = {}
            scaled_other = np.zeros(other_columns.shape)
            for i, col in enumerate(other_columns.columns):
                scalers_other[col] = MinMaxScaler(feature_range=(0, 1))
                scaled_other[:, i] = scalers_other[col].fit_transform(
                    other_columns[col].values.reshape(-1, 1)).flatten()

            scaled_data = np.concatenate((scaled_price, scaled_other), axis=1)

            self.X, self.y = self.create_sequences(scaled_data, time_step)
            input_shape = (time_step, scaled_data.shape[1])

        train_size = int(len(self.data) * argument['size'])
        self.X_train, self.X_test = self.X[:train_size], self.X[train_size:]
        self.y_train, self.y_test = self.y[:train_size], self.y[train_size:]

        self.dates = self.data.index[time_step:]
        self.train_dates, self.test_dates = self.dates[:train_size], self.dates[train_size:]

        # Start train model
        logging.info('Start train BiLSTM MODEL')
        self.create_model(argument, input_shape)
        epochs = argument['epochs']
        batchsize = argument.get('batchsize', 64)
        print(
            f"Training Parameters:\n Epochs: {epochs}\n Batch size: {batchsize}\n Time step: {time_step}\n Size: {argument['size']}\n")
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batchsize)

        print("BiLSTM model Summary:")
        self.model.summary()

        # Predict
        X_test_predict = self.model.predict(self.X_test)
        self.X_test_predict = scaler_price.inverse_transform(X_test_predict)
        self.y_test_actual = scaler_price.inverse_transform(self.y_test.reshape(-1, 1))

        # DataFrame
        self.actual_data = pd.DataFrame({
            'date': self.test_dates,
            'price': self.y_test_actual.flatten()
        }).set_index('date')

        self.forecast_data = pd.DataFrame({
            'date': self.test_dates,
            'price': self.X_test_predict.flatten()
        }).set_index('date')

        # forecast_accuracy
        self.accuracy = self.forecast_accuracy(self.y_test_actual, self.X_test_predict)

        return self.forecast_data, self.accuracy, self.model
    
    
    def ml_flow_register(self, experient_name="DEFAUT_MODEL", argument=None):
        ARTIFACT_PATH = "model"

        mlflow.set_tracking_uri(uri=self.ML_FLOW_URL)
        mlflow.set_experiment(experient_name)

        # Create an instance of a PandasDataset
        dataset = mlflow.data.from_pandas(
            self.data, source=self.data_uri, name="rice data", targets="price"
        )

        with mlflow.start_run() as run:
            input_sample = pd.DataFrame(self.train_data)
            output_sample = pd.DataFrame(self.forecast_data)

            mlflow.log_input(dataset, context="training")

            mlflow.log_params({"argument": argument})

            for k, v in self.accuracy.items():
                mlflow.log_metric(k, round(v, 4))

            signature = infer_signature(input_sample, output_sample)

            model_mflow = mlflow.sklearn.log_model(
                self.model, ARTIFACT_PATH, signature=signature
            )
            return model_mflow

if __name__ == '__main__':
    model_url = "../test_data/BiLSTM_univariate_coffee.h5"
    data_url = "../test_data/coffee_daklak.csv"
    data = pd.read_csv("../test_data/coffee_daklak.csv", encoding='utf-8')
    model = BiLSTMModel()
    model.model_url = model_url    
    model.data_url = data_url
    
    n_steps = 10
    
    # Chia tập dữ liệu
    train_data , test_data = model.prepare_data(data_url)
    
    # Dự đoán mô hình trên dữ liệu thực tế
    predicted_train_initial = model.predict(train_data, n_steps)
    predicted_test_initial = model.predict(test_data, n_steps)
    
    predicted_train = np.concatenate([train_data.iloc[:n_steps]['price'].values, predicted_train_initial.flatten()])
    predicted_test = np.concatenate([test_data.iloc[:n_steps]['price'].values, predicted_test_initial.flatten()])
    
    # # Đánh giá mô hình
    evaluation_train = model.forecast_accuracy(train_data, predicted_train)
    evaluation_test = model.forecast_accuracy(test_data, predicted_test)
    print("Evaluation on train data:", evaluation_train)
    print("Evaluation on test data:", evaluation_test)
    
    # Dự đoán giá trong tương lai
    forecast_num = 30
    predicted_df = model.forecast_future(forecast_num, test_data, n_steps)

    # Nối dữ liệu dự đoán vào tập dữ liệu gốc
    data_predicted = model.concat_dataframes(data, predicted_df)

    model.ml_flow_register()
    
    # # Vẽ biểu đồ
    # target_date = pd.Timestamp('2024-01-01')
    # model.plot_predictions(data, data_predicted, target_date)