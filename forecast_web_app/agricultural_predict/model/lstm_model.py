import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import mean_squared_error

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

def evaluate_model(test_data, predicted_values):
    mse = mean_squared_error(test_data, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_data, predicted_values)
    
    return {'rmse': rmse, 'r2': r2}

if __name__ == '__main__':
    data = pd.read_csv("../test_data/coffee.csv", encoding='utf-8')['price'].values
    n_steps = 10
    train_size = int(len(data) * 0.8) 
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    model_path = "../test_data/LSTM_univariate_coffee.h5"
    predicted_train = predict(model_path, train_data, n_steps)
    predicted_test = predict(model_path, test_data, n_steps)
    
    evaluation_train = evaluate_model(train_data[n_steps:], predicted_train)
    evaluation_test = evaluate_model(test_data[n_steps:], predicted_test)
    
    print("Evaluation on train data:", evaluation_train)
    print("Evaluation on test data:", evaluation_test)
    
    # # Plotly visualization
    # train_trace = go.Scatter(x=list(range(n_steps, n_steps + len(predicted_train))), y=predicted_train.flatten(), mode='lines', name='Predicted (Train)')
    # test_trace = go.Scatter(x=list(range(n_steps + len(train_data), n_steps + len(train_data) + len(predicted_test))), y=predicted_test.flatten(), mode='lines', name='Predicted (Test)')
    # actual_train_trace = go.Scatter(x=list(range(n_steps, n_steps + len(train_data[n_steps:]))), y=train_data[n_steps:], mode='lines', name='Actual (Train)')
    # actual_test_trace = go.Scatter(x=list(range(n_steps + len(train_data), n_steps + len(train_data) + len(test_data[n_steps:]))), y=test_data[n_steps:], mode='lines', name='Actual (Test)')
    # layout = go.Layout(title='Actual vs Predicted', xaxis=dict(title='Time'), yaxis=dict(title='Price'))
    # fig = go.Figure(data=[actual_train_trace, train_trace, test_trace, actual_test_trace], layout=layout)
    # fig.show()
    
    # Plotly visualization
    train_trace = go.Scatter(x=list(range(n_steps, n_steps + len(predicted_train))), y=predicted_train.flatten(), mode='lines', name='Predicted (Train)', hovertemplate='%{y:.2f}')
    test_trace = go.Scatter(x=list(range(n_steps + len(train_data), n_steps + len(train_data) + len(predicted_test))), y=predicted_test.flatten(), mode='lines', name='Predicted (Test)', hovertemplate='%{y:.2f}')
    
    layout = go.Layout(title='Predicted Prices', xaxis=dict(title='Time'), yaxis=dict(title='Price'))
    
    fig = go.Figure(data=[train_trace, test_trace], layout=layout)
    fig.show()