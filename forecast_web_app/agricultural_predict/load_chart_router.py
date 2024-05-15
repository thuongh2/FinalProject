import http

from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash, jsonify
from flask import current_app
from flask_cors import cross_origin
from pymongo import MongoClient
from config.db import model
from flask import session

from flask import current_app
from minio import Minio

import pandas as pd
from werkzeug.utils import secure_filename
from statsmodels.tsa.stattools import acf, pacf

from model.factory_model import FactoryModel
from statsmodels.tsa.stattools import adfuller
import io
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

load_chart_router = Blueprint('load_chart_router', __name__, static_folder='static',
            template_folder='templates')

def adf_test(series):
    result = adfuller(series.dropna())
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    current_app.logger.info(out)
    return result[1]

@load_chart_router.route('/load-chart', methods=['GET'])
@cross_origin()
def load_chart():
    model_name = request.args.get('model_name')
    model_data = request.args.get('model_data')
    model_time = request.args.get('model_time')
    model_time = int(model_time)
    n_steps = 10

    dict_model_file = {'ARIMA':'ARIMA.joblib', 'LSTM':'LSTM_univariate_coffee.h5', 'GRU':'GRU_univariate_coffee.h5', 'BiLSTM':'BiLSTM_univariate_coffee.h5'}
    model_url = './file/' + dict_model_file.get(model_name)
    model = FactoryModel(model_name).factory()
    model_url = model_url
    model.data_uri = model_data
    model.model_url = model_url
    _, test_data = model.prepare_data_for_self_train()
    predict_data = model.forecast_future(model_time, test_data, n_steps)

    predict_data['date'] = pd.to_datetime(predict_data['date'])
    predict_data.set_index(['date'], inplace=True)
    test_data = test_data.iloc[-90:]
    first_predict_data_row = predict_data.iloc[[0]]
    first_predict_data_row.index = pd.to_datetime(first_predict_data_row.index)
    test_data = pd.concat([test_data, first_predict_data_row])

    plot_data = []
    trace1 = dict(
        x=test_data.index.tolist(),
        y=test_data.price.values.tolist(),
        mode='lines',
        name='Giá thực tế',
    )
    plot_data.append(trace1)

    trace2 = dict(
        x=predict_data.index.tolist(),
        y=predict_data.price.values.tolist(),
        mode='lines',
        name='Giá dự đoán',
    )
    plot_data.append(trace2)

    response_data = {'plot_data': plot_data}
    return jsonify(response_data)

