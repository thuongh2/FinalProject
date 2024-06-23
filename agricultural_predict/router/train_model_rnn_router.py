import http

from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash, jsonify
from flask import render_template
from flask import send_from_directory
from flask import current_app
from flask_cors import cross_origin
from pymongo import MongoClient
from requests.auth import HTTPBasicAuth

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

from model.factory_model import FactoryModel
from bson import json_util
import uuid
import os
import time
from infra.airflow.include import dag_config
import requests as requests_api
from utils import common_utils, constant
from config import host_config


train_model_rnn_router = Blueprint('train_model_rnn_router', __name__, static_folder='static',
                               template_folder='templates')


@train_model_rnn_router.route('/train-model-rnn', methods=['GET'])
def train_model_rnn_page():
    model_name = request.args.get('model_name')
    model_data = model_info_collection.find()

    model_names = [m.get('name') for m in model_data]
    current_app.logger.info(model_names)
    model_id = common_utils.generate_id(model_name)
    if (model_name):
        model_data_find = model_info_collection.find_one({'name': model_name})
        data = model_data_find.get('data')
        default_param = model_data_find.get('default_param')
        params_render = None
        if default_param:
            params_render = default_param.get('param')
            current_app.logger.info(params_render)

        return render_template('admin/train-model-rnn.html',
                                model_names=model_names, data=data,
                                model_name=model_name, params_render = params_render, model_id=model_id)

    return render_template('admin/train-model-rnn.html',
                           model_names=model_names, data=None, model_name="", params_render=None)


def create_dags_flow(dags_id, model_name, user_name, data_name,
                     argument, agricultural_name, stationary_option=None):
    print(dags_id)
    is_created = dag_config.create_dags_file(dags_id, data_name, model_name, argument,
                                             agricultural_name, user_name, stationary_option)
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

    # TODO set this in os
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



@train_model_rnn_router.route('/train-model-rnn-data', methods=['POST'])
@cross_origin()
def train_model_rnn_data():
    data = request.get_json()
    model_name = data.get('model_name')
    model_data = data.get('model_data')
    argument = data.get('argument')
    model_id = data.get('model_id')

    data_model = {
                    "user_id": data.get('username'),
                    "model_name": model_name,
                    "agricultural_name": data.get('agricultural_name'),
                    "data_name": model_data,
                    "argument": argument,
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


@train_model_rnn_router.route('/submit-train-model-rnn-data', methods=['POST'])
@cross_origin()
def submit_train_model_rnn_data():
    data = request.get_json()
    if 'false' in data:
        data = (eval(data.replace("'", "\"")
                     .replace('false', 'False')
                     .replace('true', 'True')))
    else:
        data = eval(data.replace("'", "\""))
    model_traning = model_registry_collection.find_one({'_id': data.get('_id')})
    if not model_traning:
        return jsonify({'message': 'No train model found'}), 404

    return json_util.dumps(data.get('_id'))
