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
from werkzeug.utils import secure_filename
import json
from pprint import pprint
from datetime import datetime, timedelta

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
    return client.get_presigned_url(
    "GET",
    BUCKET_NAME,
    filename,
    expires=timedelta(days=1),
)
    



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
        return render_template('admin/upload-model.html', model_names=model_names, data=data, model_name=model_name)

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
    
    data_model = {"user_id": session.get('username'),
                "name": name_train,
                "model_id": model_name,
                "algricutural_name": algricutural_name,
                "data_name": data_name,
                "file_name": file_after_upload.object_name,
                "file_etag": file_after_upload.etag,
                "create_time": datetime.now()}
    train_model.insert_one(data_model)
    return redirect(url_for('upload_model_router.admin_upload_train_model'))
    
    


@upload_model_router.route('/admin')
def admin():
    train_model_list = train_model.find()
    records = list(train_model.find())
    return render_template('admin/index.html', train_model_list=train_model_list, total_model = len(records))
