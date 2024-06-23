import http
from flask import Blueprint
from flask import request, redirect, url_for, flash, jsonify
from flask_cors import cross_origin
from pymongo import MongoClient
from config.db_conn_config import model_registry_collection
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
import os
from utils import minio_utils
from infra.airflow.include import dag_config
import requests as requests_api
from utils import constant
from time import sleep
from config import host_config
from utils import common_utils

pipeline_router = Blueprint('pipeline_router',
                            __name__,
                            static_folder='static',
                            template_folder='templates')


@pipeline_router.route('/pipeline/<dags_id>/<task_id>', methods=['GET'])
@cross_origin()
def pipeline_airflow(dags_id, task_id):
    try:
        airflow_url = f"{host_config.HOST}:8080/api/v1/dags/{dags_id}/dagRuns/{task_id}/taskInstances"

        print("Get data from " + airflow_url)
        response = requests_api.get(airflow_url,
                                    auth=HTTPBasicAuth('airflow', 'airflow'),
                                    headers={'Content-Type': 'application/json'})
        return jsonify(response.text), http.HTTPStatus.OK
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), http.HTTPStatus.INTERNAL_SERVER_ERROR


@pipeline_router.route('/pipeline-logs/<dags_id>/<step>', methods=['GET'])
@cross_origin()
def pipeline_logs_airflow(dags_id, step):
    try:
        dirname = os.path.dirname(__file__)
        print(dirname)
        dag_dir = f"dag_id={dags_id}/run_id={dags_id}/task_id={step}"
        log_file = os.path.join(dirname, "infra/airflow/logs/" + dag_dir + "/attempt=1.log")

        with open(log_file) as f:
            content = f.read()
            return jsonify(content)
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), http.HTTPStatus.INTERNAL_SERVER_ERROR
    

@pipeline_router.route('/submit_auto_train_model', methods=['GET'])
@cross_origin()
def submit_auto_train_model_airflow():
    try:
        file_name = request.args.get('file_name')
        data_url = request.args.get('data_url')
        accuracy = eval(request.args.get('accuracy'))
        model_name = request.args.get('model_name')
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
        forecast_data, _ = model_factory.train_for_upload_mode(n_periods, test_data)

        if isinstance(forecast_data, pd.DataFrame):
            forecast_data.set_index(test_data.index, inplace=True)
        else:
            forecast_data = pd.DataFrame(forecast_data, index=test_data.index, columns=['price'])

        try:
            model_factory.ml_flow_register(argument=argument)
        except Exception as e:
            print(e)

        data_model = {  
                        "_id": common_utils.generate_id(model_name + 'AUTO'),
                        "user_id": constant.SYSTEM,
                        "file_name": file_name,
                        "model_name": model_name,
                        "name": f"Dự đoán giá tự động mô hình {model_name}",
                        "agricultural_name": agricultural_name,
                        "data_name": data_url,
                        "type": constant.AUTO_TRAIN_MODEL,
                        "create_time": datetime.now(),
                        "evaluate": accuracy,
                        "status": constant.SUCCESS,
                        "is_used": False,
                        "is_training": False
                    }

        model_registry_collection.insert_one(data_model)
        print("Thêm model thành công")
        return json_util.dumps(data_model), http.HTTPStatus.OK
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), http.HTTPStatus.INTERNAL_SERVER_ERROR


@pipeline_router.route('/submit_train_model', methods=['GET'])
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

        data_model = {
                        "_id": model_id,
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
        return json_util.dumps(data_model), http.HTTPStatus.OK
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), http.HTTPStatus.INTERNAL_SERVER_ERROR


@pipeline_router.route('/get_train_model_airflow/<model_id>', methods=['GET'])
@cross_origin()
def get_train_model_airflow(model_id):
    model_training = model_registry_collection.find_one({"_id": model_id})
    if not model_training:
        return json_util.dumps({"error": "Model not found"}), http.HTTPStatus.NOT_FOUND
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

    forecast_data, ac = model_factory.train_for_upload_mode(len(test_data), test_data)

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

    return json_util.dumps(data_model), http.HTTPStatus.OK