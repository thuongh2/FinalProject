import http

from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash, jsonify
from flask import render_template
from flask import send_from_directory
from flask import current_app
from flask_cors import cross_origin
from pymongo import MongoClient
from config.db import db, model, user, train_model
from flask import session
import hashlib
from flask import current_app
from minio import Minio
import numpy as np
import ast
import pandas as pd
from werkzeug.utils import secure_filename
import json
from pprint import pprint
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import acf, pacf

from model.factory_model import FactoryModel
from bson import json_util
from statsmodels.tsa.stattools import adfuller
import joblib
import uuid
import os
import utils.minio_utils as minio_utils

train_model_router = Blueprint('train_model_router', __name__, static_folder='static',
                               template_folder='templates')


@train_model_router.route('/train-model', methods=['GET'])
def train_model_page():
    model_name = request.args.get('model_name')
    model_data = model.find()

    model_names = [m.get('name') for m in model_data]
    current_app.logger.info(model_names)

    if (model_name):
        model_data_find = model.find_one({'name': model_name})
        data = model_data_find.get('attrs')
        default_param = model_data_find.get('default_param')
        if(default_param):
            params_render = default_param.get('param')
        current_app.logger.info(params_render)

        return render_template('admin/train-model.html',
                                model_names=model_names, data=data,
                                model_name=model_name, params_render = params_render)

    return render_template('admin/train-model.html',
                           model_names=model_names, data=None, model_name="", params_render=None)


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
            x=data.index.tolist(),
            y=data[columns].values.tolist(),
            mode='lines'
        )
        plot_data.append(trace)

    return plot_data


def adf_test(series):
    result = adfuller(series.dropna())
    labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
    out = pd.Series(result[0:4], index=labels)
    current_app.logger.info(out)
    return result[1]


@train_model_router.route('/stationary-train-model', methods=['GET'])
@cross_origin()
def make_stationary_data_train_model():
    model_name = request.args.get('model_name')
    model_data = request.args.get('model_data')
    is_stationary = request.args.get('is_stationary')
    diff_type = request.args.get('diff_type')
    lag = request.args.get('lag', 1)

    # diff
    data = pd.read_csv(model_data)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index(['date'], inplace=True)
    if data.empty:
        return jsonify({'error': 'Data is empty'})

    df1 = data.copy()
    if (is_stationary == 'True'):
        if (diff_type == 'log'):
            df1 = np.log(df1)
        else:
            df1 = df1 - df1.shift(int(lag))
        df1 = df1.dropna()

    # p value
    p_values = adf_test(df1.price)
    current_app.logger.info(p_values)

    # return acf pacf transformed data
    df_acf = acf(df1.price, nlags=10).tolist()

    df_pacf = pacf(df1.price, nlags=10).tolist()

    plot_data = []
    current_app.logger.info(df1.columns)
    for columns in df1.columns:
        df1[columns] = df1[columns].astype(float)
        trace = dict(
            x=df1.index.tolist(),
            y=df1[columns].values.tolist(),
            mode='lines',
            name=columns
        )
        plot_data.append(trace)

    respose_data = {'p_values': p_values, 'acf': df_acf, 'df_pacf': df_pacf, 'plot_data': plot_data}
    return jsonify(respose_data)


@train_model_router.route('/train-model-data', methods=['POST'])
@cross_origin()
def train_model_data():
    data = request.get_json()

    model_name = data.get('model_name')
    model_data = data.get('model_data')
    argument = data.get('argument')

    data_model = {"user_id": data.get('username'),
                  "model_name": model_name,
                  "agricutural_name": data.get('agricutural_name'),
                  "data_name": model_data,
                  "type": "TRAIN_MODEL",
                  "create_time": datetime.now(),
                  "isUsed": False}

    file_name = str(uuid.uuid4()) + '.joblib'
    file_dir = "./temp/" + file_name
    try:
        factory_model = FactoryModel(model_name).factory()
        factory_model.data_uri = model_data
        forecast_data, accuracy, model = factory_model.train_model(argument)

        joblib.dump(model, file_dir)

        file_after_upload = minio_utils.fupload_object(file_name,  file_dir)
        data_model["file_name"] = file_after_upload.object_name
        data_model["file_etag"] = file_after_upload.etag
        data_model['score'] = accuracy
        data_model['status'] = 'DONE'

        
        trace_predict = dict(
            x=forecast_data.index.tolist(),
            y=forecast_data.price.values.tolist(),
            mode='lines',
            name='Dự đoán'
        )
        trace_actual = dict(
            x=factory_model.test_data.index.tolist(),
            y=factory_model.test_data.price.values.tolist(),
            mode='lines',
            name='thực tế'
        )
        plot_data = [trace_predict, trace_actual]
        data_model['plot_data'] = plot_data
    except Exception as e:
        print(e)
        data_model['status'] = 'FAIL'
        data_model['error'] = str(e)

    # train_model.insert_one(data_model)
    if os.path.exists(file_dir):
        os.remove(file_dir)
        print("Remove temp file " + file_name)
    current_app.logger.info(data_model)

    return json_util.dumps(data_model)


@train_model_router.route('/submit-train-model-data', methods=['POST'])
@cross_origin()
def submit_train_model_data():
    data = request.get_json()
    data = eval(data.replace("'", "\"").replace('false', 'False'))
    
    del data['plot_data']

    train_model.insert_one(data)
    return json_util.dumps(data.get('_id'))
