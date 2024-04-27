from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash
from flask import render_template
from flask import send_from_directory
from flask import current_app
from pymongo import MongoClient
from config.db import db, model, user, train_model
from flask import session
import hashlib
from flask import current_app
from minio import Minio
import os
import pandas as pd
from werkzeug.utils import secure_filename
import json
from pprint import pprint
from datetime import datetime, timedelta

from model.factory_model import FactoryModel
from model.arima_model import ARIMAModel
from bson.objectid import ObjectId
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import ast


upload_model_router = Blueprint('upload_model_router', __name__, static_folder='static',
            template_folder='templates')

BUCKET_NAME = 'test'
MINIO_URL = '20.2.210.176:9000'
MINIO_ACCESS_KEY = 'minio'
MINIO_SECRET = 'minio123'


def upload_object(filename, data, length):
    client = Minio(MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET, secure=False)

    # Make bucket if not exist.
    found = client.bucket_exists(BUCKET_NAME)
    if not found:
        client.make_bucket(BUCKET_NAME)
    else:
        print(f"Bucket {BUCKET_NAME} already exists")

    file = client.put_object(BUCKET_NAME, filename, data, length)
    print(f"{filename} is successfully uploaded to bucket {BUCKET_NAME}.")
    return file



def get_minio_object(filename):
    client = Minio(MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET, secure=False)
    client.fget_object(BUCKET_NAME, filename, "./file/" + filename)
    return "./file/" + filename
    




@upload_model_router.route('/upload-model-minio', methods=['GET', 'POST'])
def upload_model():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            return redirect(request.url)
        if file:
            size = os.fstat(file.fileno()).st_size
            file_after_upload = upload_object(file.filename, file, size)
            current_app.logger.info(file_after_upload.__dir__())
            current_app.logger.info(file_after_upload.object_name)
            current_app.logger.info(file_after_upload.version_id)
            current_app.logger.info(file_after_upload.etag)
            return redirect(request.url)
    return render_template('admin/upload-model.html')


@upload_model_router.route('/minio', methods=['GET'])
def get_minio_model():
    file_name = request.args.get('file_name')
    # Get data of an object.
    return get_minio_object(file_name)


     

@upload_model_router.route('/upload-model')
def admin_upload_train_model():
    model_name = request.args.get('model_name')
    model_data = model.find()

    model_names = [m.get('name') for m in model_data]
   
    if(model_name):
        model_data_find = model.find_one({'name': model_name})
        data = model_data_find.get('attrs')
        data_url = []

        for item in data.get('data'):
            data_url.append(list(item.values())[0])

        current_app.logger.info(data_url)
    
        return render_template('admin/upload-model.html', model_names=model_names, data=data, model_name=model_name, data_url=data_url)

    return render_template('admin/upload-model.html', 
                           model_names=model_names, data=None, model_name="")


@upload_model_router.route('/upload-model', methods=['POST'])
def admin_train_model():
    name_train = request.form['name_train']
    model_name = request.args.get('model_name')
    algricutural_name = request.form['algricutural_name']
    data_name = request.form['data_name']

    file = request.files["file"]
    # If the user does not select a file, the browser submits an
     # empty file without a filename.
    if file.filename == "":
        current_app.logger.info("KHÔNG TRỐNG")
        return redirect(request.url)
    if file:
        size = os.fstat(file.fileno()).st_size
        file_after_upload = upload_object(file.filename, file, size)
        current_app.logger.info(file_after_upload.__dir__())
        current_app.logger.info(file_after_upload.object_name)
        current_app.logger.info(file_after_upload.version_id)
        current_app.logger.info(file_after_upload.etag)
        
    # kiểm tra model

    arima_model = FactoryModel(model_name).factory()
    model_url = get_minio_object(file_after_upload.object_name)
    current_app.logger.info(model_url)
    arima_model.model_url = model_url
    arima_model.data_uri = data_name
    _, test_data = arima_model.prepare_data(arima_model.data_uri)
    # xử lí dữ liệu (cho trai trên web)
    current_app.logger.info(test_data.head())
    data, ac = arima_model.train_for_upload_mode(len(test_data), test_data)
    current_app.logger.info(ac)

    
    data_model = {"user_id": session.get('username'),
                "name": name_train,
                "model_id": model_name,
                "algricutural_name": algricutural_name,
                "data_name": data_name,
                "file_name": file_after_upload.object_name,
                "file_etag": file_after_upload.etag,
                "create_time": datetime.now(), 
                "score": ac}
    train_model.insert_one(data_model)
    return redirect("detail-model?model_id=" + str(data_model.get('_id')))
    
    


def create_chart_mode(data_actual, data_predicted, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_actual.index, y=data_actual['price'], mode='lines', name='Giá thực tế',
                             line=dict(color='rgba(0, 0, 255, 0.5)'), fill='tozeroy', fillcolor='rgba(173, 216, 230, 0.2)', visible=True))
    fig.add_trace(go.Scatter(x=data_predicted.index, y=data_predicted['price'], mode='lines',
                             name='Giá dự đoán', line=dict(color='rgba(255, 165, 0, 0.5)'), fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.2)', visible=True))
    fig.update_layout(
                    xaxis_title='Ngày',
                    yaxis_title='Giá',
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                       tickmode='array',
                       dtick='7D', 
                       tickformat='%d-%m-%Y' 
                   ))
    # fig.show()
    
    pio.write_html(fig, './templates/chart/' + model_name + '.html')

@upload_model_router.route('/detail-model')
def admin_detail_model():
    model_id = request.args.get('model_id')
    current_app.logger.info(model_id)
    model_data = train_model.find_one(ObjectId(model_id))
    current_app.logger.info(model_data)
    # load model
    if model_data:
        arima_model = FactoryModel(model_data.get('model_id')).factory()
        model_url = get_minio_object(model_data.get('file_name'))
        current_app.logger.info(model_url)
        arima_model.model_url = model_url
        arima_model.data_uri = model_data.get('data_name')
        _ , test_data = arima_model.prepare_data(arima_model.data_uri)
        # xử lí dữ liệu (cho trai trên web)
        current_app.logger.info(test_data.head())
        data , ac = arima_model.train_for_upload_mode(len(test_data), test_data)
        current_app.logger.info(ac)
        if(isinstance(data, pd.DataFrame)):
            data.set_index(test_data.index, inplace=True)
        else:
            data = pd.DataFrame(data, index=test_data.index, columns=['price'])
        current_app.logger.info(test_data.head())

        current_app.logger.info(data.head())
        create_chart_mode(test_data, data, model_data.get('model_id') + str(model_data.get('_id')))
    else:
        flash('Không tìm thấy mô hình.', 'danger')
    return render_template('admin/detail-model.html', model_data= model_data, chart_name = model_data.get('model_id') + str(model_data.get('_id')))


@upload_model_router.route('/admin')
def admin():
    train_model_list = train_model.find()
    records = list(train_model.find())
    return render_template('admin/index.html', train_model_list=train_model_list, total_model = len(records))
