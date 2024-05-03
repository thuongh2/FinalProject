import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error

class BiLSTMModel():
    def __init__(self):
        # model
        self.model = None
        # link model
        self.model_url = ""
        # link dữ liệu
        self.data_url = ""
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
        
    def prepare_data(self, train_url, test_url):
        if train_url:
            self.train_data = pd.read_csv(train_url)
        if test_url:
            self.test_data = pd.read_csv(test_url)
            
        return self.set_index_date(self.train_data, self.test_data)
    
    
    def prepare_data(self, data_url):
        self.data = pd.read_csv(data_url)
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

    def forecast_accuracy(self, test_data, predicted_values):
        test_values = test_data['price'].values
        mape = np.mean(np.abs((test_values - predicted_values) / test_values)) * 100
        mse = mean_squared_error(test_values, predicted_values)
        rmse = np.sqrt(mse)

        return {'mape': round(mape, 2), 'rmse': round(rmse, 2)}

    
    def predict_ensemble(self, forecast_num, data, n_steps, time):
        model = load_model(self.model_url)
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices = data['price'].values
        dataset = scaler.fit_transform(prices.reshape(-1, 1))
        last_data = dataset[-time:]
        last_data = last_data.reshape(1, -1)[:, -(time-1):]
    
        predicted_prices = []
        for day in range(forecast_num):
            next_prediction = model.predict(last_data)
            last_data = np.append(last_data, next_prediction).reshape(1, -1)[:, 1:]
            predicted_price = scaler.inverse_transform(next_prediction.reshape(-1, 1))
            predicted_prices.append(predicted_price[0, 0])
    
        return predicted_prices
    
    def plot_predictions(self, data, data_predicted, target_date):
        last_date = data['date'].iloc[-1]
        list_predicted = data_predicted[pd.to_datetime(data_predicted['date']) >= last_date]
        data_actual = data[pd.to_datetime(data['date']) >= target_date] 

    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data_actual['date'], y=data_actual['price'], mode='lines', name='Giá thực tế', line=dict(color='rgba(0, 0, 255, 0.5)'), fill='tozeroy', fillcolor='rgba(173, 216, 230, 0.7)', visible=True))
        fig.add_trace(go.Scatter(x=list_predicted['date'], y=list_predicted['price'], mode='lines', name='Giá dự đoán', line=dict(color='rgba(255, 165, 0, 0.5)'), fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.2)', visible=True))
        fig.update_layout(
                        title={
                            'text': "BIỂU ĐỒ DỰ ĐOÁN GIÁ CÀ PHÊ",
                            'font': {
                                'family': 'Arial',
                                'size': 20,
                            }
                        },
                        title_x=0.5,
                        xaxis_title='Ngày',
                        yaxis_title='Giá',                       
                        yaxis=dict(range=[52000, max(data_predicted['price'])]),
                        plot_bgcolor='rgba(0,0,0,0)', 
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(  
                            tickformat='%d/%m/%Y',                     
                            tickvals=data_predicted['date'][::14],                                 
                    ))
        
        pio.write_html(fig, '../templates/chart/bilstm_univariate_coffee_60days.html')


    def train_for_upload_mode(self, n_periods, test_data):
        n_steps = 10
        forecast = self.predict(test_data, n_steps)
        forecast = np.concatenate([test_data.iloc[:n_steps]['price'].values, forecast.flatten()])
        self.forecast_data = pd.DataFrame(forecast, columns=['price'])
        self.forecast_data.set_index(test_data.index, inplace=True)

        if self.forecast_data.empty:
            raise Exception("Không tìm thấy model")
        print(self.forecast_data.info())

        self.accuracy = self.forecast_accuracy(self.test_data.price.values, self.forecast_data.price.values)
        return self.forecast_data, self.accuracy

    def forecast_future(self, forecast_num, data, n_steps):
        predicted = self.predict_ensemble(forecast_num, data, n_steps, n_steps+1)
        last_date = data.index[-1]
        next_dates = pd.date_range(start=last_date, periods=forecast_num + 1)[1:]
        predicted_df = pd.DataFrame({'date': next_dates, 'price': predicted})
        
        return predicted_df
    
    def concat_dataframes(self, original_df, predicted_df):
        predicted_df['date'] = pd.to_datetime(predicted_df['date'])
        predicted_df['date'] = predicted_df['date'].dt.strftime('%m/%d/%Y')
        original_df['date'] = pd.to_datetime(original_df['date'])
        original_df['date'] = original_df['date'].dt.strftime('%m/%d/%Y')
        data_predicted = pd.concat([original_df, predicted_df], ignore_index=True)
        
        return data_predicted


    
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
    forecast_num = 60
    predicted_df = model.forecast_future(forecast_num, test_data, n_steps)

    # Nối dữ liệu dự đoán vào tập dữ liệu gốc
    data_predicted = model.concat_dataframes(data, predicted_df)
    
    # Vẽ biểu đồ
    target_date = pd.Timestamp('2024-01-01')
    model.plot_predictions(data, data_predicted, target_date)