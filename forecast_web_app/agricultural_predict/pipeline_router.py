import http

from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash, jsonify, Response, stream_with_context
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
from requests.auth import HTTPBasicAuth
from model.factory_model import FactoryModel
from bson import json_util
from statsmodels.tsa.stattools import adfuller
import joblib
import uuid
import os
import utils.minio_utils as minio_utils
from infra.airflow.include import dag_config
import requests as requests_api
import time
from utils import constant
from time import sleep

pipeline_router = Blueprint('pipeline_router', __name__, static_folder='static',
                               template_folder='templates')

@pipeline_router.route('/pipeline/<dags_id>/<task_id>', methods=['GET'])
@cross_origin()
def pipeline_airflow(dags_id, task_id):
    try:
        airflow_url = f"http://localhost:8080/api/v1/dags/{dags_id}/dagRuns/{task_id}/taskInstances"
        print("Get data from " + airflow_url)
        response = requests_api.get(airflow_url, 
                                    auth=HTTPBasicAuth('airflow', 'airflow'),
                                    headers={'Content-Type': 'application/json'})
        return jsonify(response.text)
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), http.HTTPStatus.INTERNAL_SERVER_ERROR


@pipeline_router.route('/pipeline-logs/<dags_id>/<step>', methods=['GET'])
@cross_origin()
def pipeline_logs_airflow(dags_id, step):
    try:
        dirname = os.path.dirname(__file__)
        print(dirname)
        dag_dir = "dag_id=" + dags_id + "/run_id=" + dags_id + "/" + "task_id=" + step
        log_file = os.path.join(dirname, "infra/airflow/logs/" + dag_dir + "/attempt=1.log")

        with open(log_file) as f:
            content = f.read()
            return jsonify(content)
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), http.HTTPStatus.INTERNAL_SERVER_ERROR