import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error

# Dự đoán giá trên dữ liệu có sẵn
def predict(model_path, test_data, n_steps):
    model = load_model(model_path)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_test_data = scaler.fit_transform(test_data.reshape(-1, 1))
    X_test = []
    for i in range(n_steps, len(scaled_test_data)):
        X_test.append(scaled_test_data[i-n_steps:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_values = model.predict(X_test)
    predicted_values = scaler.inverse_transform(predicted_values)
    
    return predicted_values

# Đánh giá mô hình
def evaluate_model(test_data, predicted_values):
    mse = mean_squared_error(test_data, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_data, predicted_values)
    
    return {'rmse': rmse, 'r2': r2}

# Dự đoán giá trong tương lai
def predict_ensemble(forecast_num, model, data, n_steps, time):

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data.reshape(-1, 1))
    
    last_data = dataset[-time:]
    last_data = last_data.reshape(1, -1)[:, -(time-1):]

    predicted_prices = []

    for day in range(forecast_num):
        next_prediction = model.predict(last_data)
        last_data = np.append(last_data, next_prediction).reshape(1, -1)[:, 1:]
        predicted_price = scaler.inverse_transform(next_prediction.reshape(-1, 1))
        predicted_prices.append(predicted_price[0, 0])

    return predicted_prices

if __name__ == '__main__':
    data = pd.read_csv("../test_data/coffee.csv", encoding='utf-8')
    dates = pd.to_datetime(data['date'])
    prices = data['price'].values
    
    # Chia tập dữ liệu
    n_steps = 10
    train_size = int(len(prices) * 0.8) 
    train_data = prices[:train_size]
    test_data = prices[train_size:]
    
    # Dự đoán mô hình trên dữ liệu thực tế
    model_path = "../test_data/LSTM_univariate_coffee.h5"
    predicted_train = predict(model_path, train_data, n_steps)
    predicted_test = predict(model_path, test_data, n_steps)
    
    # Đánh giá mô hình
    evaluation_train = evaluate_model(train_data[n_steps:], predicted_train)
    evaluation_test = evaluate_model(test_data[n_steps:], predicted_test)
    print("Evaluation on train data:", evaluation_train)
    print("Evaluation on test data:", evaluation_test)
    
    # Tạo một df và dự đoán giá trong tương lai
    forecast_num = 30
    predicted = predict_ensemble(forecast_num, load_model(model_path), test_data, n_steps, n_steps+1)
    last_date = dates.iloc[int(len(data)) - 1]  
    next_dates = pd.date_range(start=last_date, periods=forecast_num + 1)[1:]  
    predicted_df = pd.DataFrame({'date': next_dates, 'price': predicted})
    
    data_predicted = pd.concat([data, predicted_df], ignore_index=True)

    
    # Vẽ biểu đồ
    target_date = pd.Timestamp('2024-02-01')
    list_predicted = data_predicted[pd.to_datetime(data_predicted['date']) >= last_date]
    data_actual = data[pd.to_datetime(data['date']) >= target_date]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_actual['date'], y=data_actual['price'], mode='lines', name='Giá thực tế', line=dict(color='rgba(0, 0, 255, 0.5)'), fill='tozeroy', fillcolor='rgba(173, 216, 230, 0.2)', visible=True))
    fig.add_trace(go.Scatter(x=list_predicted['date'], y=list_predicted['price'], mode='lines', name='Giá dự đoán', line=dict(color='rgba(255, 165, 0, 0.5)'), fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.2)', visible=True))
    fig.update_layout(title='BIỂU ĐỒ DỰ ĐOÁN GIÁ CÀ PHÊ',
                    title_font=dict(size=20),
                    title_x=0.5,
                    xaxis_title='Ngày',
                    yaxis_title='Giá',
                    yaxis=dict(range=[53500, max(data_predicted['price'])]),
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                       tickmode='array',
                       dtick='7D', 
                       tickformat='%d-%m-%Y' 
                   ))
    # fig.show()
    
    pio.write_html(fig, '../templates/chart/lstm_chart.html')