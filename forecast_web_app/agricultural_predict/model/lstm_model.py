import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from model.base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging

class LSTMModel(BaseModel):
    def __init__(self):
        super().__init__()

    def predict(self, data, n_steps):
        model = load_model(self.model_url)
        
        prices = data['price'].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices.reshape(-1, 1))
        X = []
        for i in range(n_steps, len(scaled_data)):
            X.append(scaled_data[i-n_steps:i, 0])
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        predicted_values = model.predict(X)
        predicted_values = scaler.inverse_transform(predicted_values)
        
        return predicted_values

    # def forecast_accuracy(self, test_data, predicted_values):
    #     test_values = self.test_data['price'].values
    #     mape = np.mean(np.abs((test_values - predicted_values) / test_values)) * 100
    #     mse = mean_squared_error(test_values, predicted_values)
    #     rmse = np.sqrt(mse)
    #
    #     return {'mape': round(mape, 2), 'rmse': round(rmse, 2)}

    def forecast_future(self, forecast_num, data, n_steps):
        self.model = load_model(self.model_url)
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices = data['price'].values
        dataset = scaler.fit_transform(prices.reshape(-1, 1))
        last_data = dataset[-(n_steps+1):]
        last_data = last_data.reshape(1, -1)[:, -((n_steps+1) - 1):]

        predicted_prices = []
        for day in range(forecast_num):
            next_prediction = self.model.predict(last_data)
            last_data = np.append(last_data, next_prediction).reshape(1, -1)[:, 1:]
            predicted_price = scaler.inverse_transform(next_prediction.reshape(-1, 1))
            predicted_prices.append(predicted_price[0, 0])

        last_date = self.data.index[-1]
        next_dates = pd.date_range(start=last_date, periods=forecast_num + 1)[1:]
        predicted_df = pd.DataFrame({'date': next_dates, 'price': predicted_prices})

        return predicted_df

    def train_for_upload_mode(self, n_periods, test_data):
        n_steps = 10
        forecast = self.predict(test_data, n_steps)
        forecast = np.concatenate([test_data.iloc[:n_steps]['price'].values, forecast.flatten()])
        self.forecast_data = pd.DataFrame(forecast, columns=['price'])
        self.forecast_data.set_index(test_data.index, inplace=True)

        if self.forecast_data.empty:
            raise Exception("Không tìm thấy model")
        print(self.forecast_data.info())

        self.accuracy = self.forecast_accuracy(self.test_data, self.forecast_data.price.values)
        return self.forecast_data, self.accuracy

    def concat_dataframes(self, original_df, predicted_df):
        predicted_df['date'] = pd.to_datetime(predicted_df['date'])
        predicted_df['date'] = predicted_df['date'].dt.strftime('%m/%d/%Y')
        original_df['date'] = pd.to_datetime(original_df['date'])
        original_df['date'] = original_df['date'].dt.strftime('%m/%d/%Y')
        data_predicted = pd.concat([original_df, predicted_df], ignore_index=True)

        return data_predicted

    def train_model(self, argument):
        """
        Train model in web
        params argument
            size: size split data
            layer: list of layer configurations
            epoch: number of epochs
            batch_size: size of the batch
        return LSTM MODEL
        """
        def create_sequences(data, time_step):
            X, y = [], []
            for i in range(len(data) - time_step):
                X.append(data[i:(i + time_step)])
                y.append(data[i + time_step])
            return np.array(X), np.array(y)

        # Prepare data
        if argument.get('size', 0.8) is None:
            raise Exception("Size is required")

        self.data = pd.read_csv(self.data_uri)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_price = scaler.fit_transform(self.data[self.PRICE_COLUMN].values.reshape(-1, 1))

        time_step = 10
        self.X, self.y = create_sequences(scaled_price, time_step)

        train_size = int(len(self.data) * argument['size'])
        self.X_train, self.X_test = self.X[:train_size], self.X[train_size:]
        self.y_train, self.y_test = self.y[:train_size], self.y[train_size:]

        logging.info('Start train LSTM MODEL')

        model = Sequential()
        layers_data = argument.get('layers_data', [{'id': 0, 'units': 64}])

        if len(layers_data) == 1:
            units = layers_data[0]['units']
            model.add(LSTM(units, return_sequences=False, input_shape=(time_step, 1)))
            model.add(Dropout(0.2))
        else:
            for i, layer in enumerate(layers_data):
                units = layer['units']
                if i == 0:
                    model.add(LSTM(units, return_sequences=True, input_shape=(time_step, 1)))
                    model.add(Dropout(0.2))
                elif i == len(layers_data) - 1:
                    model.add(LSTM(units, return_sequences=False))
                    model.add(Dropout(0.2))
                else:
                    model.add(LSTM(units, return_sequences=True))
                    model.add(Dropout(0.2))
        model.add(Dense(1))

        optimizer = Adam()
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        epochs = argument['epochs']
        batchsize = argument.get('batchsize', 64)
        model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batchsize)
        
        self.model = model

        X_test_predict = model.predict(self.X_test)
        X_test_predict = scaler.inverse_transform(X_test_predict)

        y_test_predict = scaler.inverse_transform(self.y_test)

        self.accuracy = self.forecast_accuracy(X_test_predict, y_test_predict)

        return self.accuracy, self.model


if __name__ == '__main__':
    model_url = "../test_data/LSTM_univariate_coffee.h5"
    data_url = "../test_data/coffee_daklak.csv"
    # data = pd.read_csv("../test_data/coffee_daklak.csv", encoding='utf-8')
    model = LSTMModel()
    model.model_url = model_url
    model.data_uri = data_url
    #
    # n_steps = 10
    #
    # # Chia tập dữ liệu
    train_data, test_data = model.prepare_data(data_url)
    #
    # # Dự đoán mô hình trên dữ liệu thực tế
    # predicted_train_initial = model.predict(train_data, n_steps)
    # predicted_test_initial = model.predict(test_data, n_steps)
    #
    # predicted_train = np.concatenate([train_data.iloc[:n_steps]['price'].values, predicted_train_initial.flatten()])
    # predicted_test = np.concatenate([test_data.iloc[:n_steps]['price'].values, predicted_test_initial.flatten()])
    #
    # # Đánh giá mô hình
    # evaluation_train = model.forecast_accuracy(train_data, predicted_train)
    evaluation_test = model.forecast_future(10, test_data, 10)
    # print("Evaluation on train data:", evaluation_train)
    print("Evaluation on test data:", evaluation_test)
    #
    # # Dự đoán giá trong tương lai
    # forecast_num = 60
    # predicted_df = model.forecast_future(forecast_num, test_data, n_steps)
    #
    # # Nối dữ liệu dự đoán vào tập dữ liệu gốc
    # data_predicted = model.concat_dataframes(data, predicted_df)
    #
    # # # Vẽ biểu đồ
    # # target_date = pd.Timestamp('2024-01-01')
    # # model.plot_predictions(data, data_predicted, target_date)
