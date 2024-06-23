from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash, jsonify
from flask import render_template
from flask import send_from_directory
from flask import current_app
from flask_cors import cross_origin
from pymongo import MongoClient
from config.db_conn_config import db, model_info_collection, user_collection, model_registry_collection
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
from requests.auth import HTTPBasicAuth
from model.factory_model import FactoryModel
from bson import json_util
from statsmodels.tsa.stattools import adfuller
import joblib
import uuid
import os
from utils import minio_utils
from infra.airflow.include import dag_config
import requests as requests_api
import time
from utils import constant
from utils import common_utils
from config import host_config
import http

train_model_router = Blueprint('train_model_router',
                               __name__,
                               static_folder='static',
                               template_folder='templates')


@train_model_router.route('/train-model', methods=['GET'])
def train_model_page():
    model_name = request.args.get('model_name')
    model_data = model_info_collection.find()

    model_names = [m.get('name') for m in model_data]
    current_app.logger.info(model_names)

    if (model_name):
        model_entity = model_info_collection.find_one({'name': model_name})
        data = model_entity.get('data')
        default_param = model_entity.get('default_param')

        if (default_param):
            params_render = default_param.get('param')
            stationary_option = default_param.get('stationary_option')

        model_id = common_utils.generate_id(model_name)

        return render_template('admin/train-model.html',
                               model_names=model_names,
                               data=data,
                               model_name=model_name,
                               params_render=params_render,
                               stationary_option=stationary_option,
                               model_id=model_id)

    return render_template('admin/train-model.html',
                           model_names=model_names,
                           data=None,
                           model_name="",
                           params_render=None,
                           stationary_option=None)


@train_model_router.route('/get-data-self-train', methods=['GET'])
def get_data_train_model():
    model_data = request.args.get('model_data')
    smoothing_type = request.args.get('smoothing_type')
    smoothing_value = request.args.get('smoothing_value')

    base_model = FactoryModel(constant.BASE_MODEL).factory()
    base_model.data_uri = model_data
    base_model.prepare_data_for_self_train()

    if base_model.data.empty:
        return jsonify({'error': 'Data is empty'})
    if smoothing_type:
        base_model.smoothing_data(type=smoothing_type, smoothing_value=smoothing_value)
    data = base_model.data
    plot_data = []
    for columns in data.columns:
        data[columns] = data[columns].astype(float)

        trace = dict(
            x=data.index.tolist(),
            y=data[columns].values.tolist(),
            mode='lines'
        )
        plot_data.append(trace)

    return plot_data, http.HTTPStatus.OK


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

    response_data = {'p_values': p_values, 'acf': df_acf, 'df_pacf': df_pacf, 'plot_data': plot_data}
    return jsonify(response_data)


def create_dags_flow(dags_id, model_name, user_name, data_name,
                     argument, agricultural_name, stationary_option=None):
    print(dags_id)
    is_created = dag_config.create_dags_file(dags_id, data_name, model_name, argument,
                                             agricultural_name,user_name, stationary_option)
    if not is_created:
        raise Exception("DAG creation failed")

    return start_trigger_airflow(dags_id)


def start_trigger_airflow(dag_id, retry=None):
    while (True):
        if (dag_config.check_dag_exists(dag_id)):
            break

    print("Waiting for setup pipeline scheduling airflow")
    time.sleep(10)
    print("Start airflow trigger")

    airflow_url = f"{host_config.HOST}:8080/api/v1/dags/{dag_id}/dagRuns"
    airflow_url = airflow_url.replace("{dag_id}", dag_id)
    print("Start trigger " + airflow_url)
    dags_run_id = dag_id + "_" + common_utils.generate_string()
    response = requests_api.post(airflow_url,
                                 auth=HTTPBasicAuth('airflow', 'airflow'),
                                 headers={'Content-Type': 'application/json'},
                                 data=json.dumps({
                                     "dag_run_id": dags_run_id
                                 }))
    if (response.status_code >= 404):
        if retry is None:
            retry = 0
        retry = retry + 1
        if retry >= 3:
            return
        print(f"Retry airflow dag run {retry}")
        return start_trigger_airflow(dag_id, retry)
    print(response.json())
    return dags_run_id


@train_model_router.route('/train-model-data', methods=['POST'])
@cross_origin()
def train_model_data():
    data = request.get_json()

    model_name = data.get('model_name')
    model_data = data.get('model_data')
    model_id = data.get('model_id')
    argument = data.get('argument')

    data_model = {
        "_id": model_id,
        "user_id": data.get('username'),
        "model_name": model_name,
        "agricultural_name": data.get('agricultural_name'),
        "data_name": model_data,
        "type": "TRAIN_MODEL",
        "create_time": datetime.now(),
        "is_used": False
    }

    dag_run_id = create_dags_flow(model_id, model_name, data_model.get('user_id'),
                                  model_data, argument, data.get('agricultural_name'),
                                  argument.get("stationary_type"))

    current_app.logger.info(data_model)
    data_model['dag_run_id'] = dag_run_id

    return json_util.dumps(data_model)


@train_model_router.route('/submit-train-model-data', methods=['POST'])
@cross_origin()
def submit_train_model_data():
    data = request.get_json()
    if 'false' in data:
        data = (eval(data.replace("'", "\"")
                     .replace('false', 'False')
                     .replace('true', 'True')))
    else:
        data = eval(data.replace("'", "\""))
    model_tranning = model_registry_collection.find_one({'_id': data.get('_id')})
    if not model_tranning:
        return jsonify({'message': 'No train model found'}), 404

    return json_util.dumps(data.get('_id'))


@train_model_router.route('/submit_train_model', methods=['GET'])
@cross_origin()
def submit_train_model_airflow():
    try:
        file_name = request.args.get('file_name')
        data_url = request.args.get('data_url')
        accuracy = eval(request.args.get('accuracy'))
        model_name = request.args.get('model_name')
        user_name = request.args.get('user_name')
        agricultural_name = request.args.get('agricultural_name')
        model_id = request.args.get('model_id')
        argument = eval(request.args.get('argument'))

        model_url = minio_utils.get_minio_object(file_name)
        model_factory = FactoryModel(model_name).factory()
        model_factory.model_url = model_url
        model_factory.data_uri = data_url
        model_factory.accuracy = accuracy

        model_factory.load_model()
        _, test_data = model_factory.prepare_data_for_self_train()

        n_periods = len(test_data)
        forecast_data, ac = model_factory.train_for_upload_mode(n_periods, test_data)

        if isinstance(forecast_data, pd.DataFrame):
            forecast_data.set_index(test_data.index, inplace=True)
        else:
            forecast_data = pd.DataFrame(forecast_data, index=test_data.index, columns=['price'])

        try:
            model_factory.ml_flow_register(argument=argument)
        except Exception as e:
            print(e)

        data_model = {"_id": model_id,
                    "user_id": user_name,
                    "file_name": file_name,
                    "model_name": model_name,
                    "name": f"Dự đoán giá mô hình {model_name}",
                    "agricultural_name": agricultural_name,
                    "data_name": data_url,
                    "type": constant.SELF_TRAIN_MODEL,
                    "create_time": datetime.now(),
                    "evaluate": accuracy,
                    "status": constant.SUCCESS,
                    "is_used": False,
                    "is_training": False
                    }

        model_registry_collection.insert_one(data_model)
        print("Thêm model thành công")

        trace_predict = dict(
            x=forecast_data.index.tolist(),
            y=forecast_data.price.values.tolist(),
            mode='lines',
            name='Dự đoán'
        )
        trace_actual = dict(
            x=test_data.index.tolist(),
            y=test_data.price.values.tolist(),
            mode='lines',
            name='thực tế'
        )
        plot_data = [trace_predict, trace_actual]
        data_model['plot_data'] = plot_data
        return json_util.dumps(data_model), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@train_model_router.route('/get_train_model_airflow/<model_id>', methods=['GET'])
@cross_origin()
def get_train_model_airflow(model_id):
    model_training = model_registry_collection.find_one({"_id": model_id})
    if not model_training:
        return json_util.dumps({"error": "Model not found"}), 404
    model_name = model_training.get('model_name')
    model_url = minio_utils.get_minio_object(model_training.get('file_name'))
    data = pd.read_csv(model_training.get('data_name'))
    model_factory = FactoryModel(model_name).factory()
    model_factory.model_url = model_url
    model_factory.data_uri = model_training.get('data_name')
    model_factory.data = data
    model_factory.accuracy = model_training.get('evaluate')

    model_factory.load_model()
    _, test_data = model_factory.prepare_data_for_self_train()

    n_periods = len(test_data)
    forecast_data, ac = model_factory.train_for_upload_mode(n_periods, test_data)

    if isinstance(forecast_data, pd.DataFrame):
        forecast_data.set_index(test_data.index, inplace=True)
    else:
        forecast_data = pd.DataFrame(forecast_data, index=test_data.index, columns=['price'])

    data_model = model_training

    trace_predict = dict(
        x=forecast_data.index.tolist(),
        y=forecast_data.price.values.tolist(),
        mode='lines',
        name='Dự đoán'
    )
    trace_actual = dict(
        x=test_data.index.tolist(),
        y=test_data.price.values.tolist(),
        mode='lines',
        name='thực tế'
    )
    plot_data = [trace_predict, trace_actual]
    data_model['plot_data'] = plot_data

    return json_util.dumps(data_model)