import http

from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash, jsonify
from flask import render_template
from flask import send_from_directory
from flask import current_app
from pymongo import MongoClient
from config.db import db, model, user, train_model
from flask import session
import hashlib
from flask import current_app
from minio import Minio
import numpy as np
import os
import pandas as pd
from werkzeug.utils import secure_filename
import json
from pprint import pprint
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import acf, pacf

from model.factory_model import FactoryModel
from model.arima_model import ARIMAModel
from bson.objectid import ObjectId
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import ast
from statsmodels.tsa.stattools import adfuller


train_model_router = Blueprint('train_model_router', __name__, static_folder='static',
            template_folder='templates')


@train_model_router.route('/train-model', methods=['GET'])
def train_model():
    current_app.logger.info("VO")
    model_name = request.args.get('model_name')
    model_data = model.find()

    model_names = [m.get('name') for m in model_data]
    current_app.logger.info(model_names)
   
    if(model_name):
        model_data_find = model.find_one({'name': model_name})
        data = model_data_find.get('attrs')
    
        return render_template('admin/train-model.html', model_names=model_names, data=data, model_name=model_name
                               )

    return render_template('admin/train-model.html', 
                           model_names=model_names, data=None, model_name="")



@train_model_router.route('/search-train-model', methods=['GET'])
def get_data_train_model():
    model_data = request.args.get('model_data')

    data = pd.read_csv(model_data)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index(['date'], inplace=True)
    if data.empty:
        return jsonify({'error': 'Data is empty'})
    plot_data = []
    for columns in data.columns:
        data[columns] = data[columns].astype(float)
       
        trace = dict(
            x = data.index.tolist(),
            y = data[columns].values.tolist(),
            mode = 'lines'
        )
        plot_data.append(trace)

    return plot_data


def adf_test(series):
    result = adfuller(series.dropna())
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)
    current_app.logger.info(out)
    return result[1]

@train_model_router.route('/stationary-train-model', methods=['GET'])
def make_stationary_data_train_model():
    model_name = request.args.get('model_name')
    model_data = request.args.get('model_data')
    is_stationary = request.args.get('is_stationary')
    diff_type = request.args.get('diff_type')
    lag = request.args.get('lag')

    # diff
    data = pd.read_csv(model_data)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index(['date'], inplace=True)
    if data.empty:
        return jsonify({'error': 'Data is empty'})


    df1 = data.copy()
    if(is_stationary == 'True'):
        if(diff_type == 'log'):
            df1 = np.log(df1)
        else:
            if not lag:
                lag = 1
            df1 = df1 - df1.shift(int(lag))
        df1 = df1.dropna()
    current_app.logger.info(df1.head())
    # p value
    p_values = adf_test(df1.price)
    current_app.logger.info(p_values)

    # return acf pacf transformed data
    df_acf =  acf(df1.price, nlags=10).tolist()

    df_pacf =  pacf(df1.price, nlags=10).tolist()

    plot_data = []
    current_app.logger.info(df1.columns)
    for columns in df1.columns:
        df1[columns] = df1[columns].astype(float)
       
        trace = dict(
            x = df1.index.tolist(),
            y = df1[columns].values.tolist(),
            mode = 'lines',
            name = columns
        )
        plot_data.append(trace)
    
    respose_data = {'p_values': p_values, 'acf': df_acf, 'df_pacf': df_pacf, 'plot_data': plot_data}
    return jsonify(respose_data)
