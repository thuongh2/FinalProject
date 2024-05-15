from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash, jsonify
from flask import render_template
from config.db import db, model, user, train_model
from flask import session
from flask import current_app
from minio import Minio
import os
import pandas as pd
from datetime import datetime, timedelta

from model.factory_model import FactoryModel
from bson.objectid import ObjectId
import plotly.graph_objs as go
import plotly.io as pio
import ast
import utils.constant as constant
import utils.minio_utils as minio_utils

upload_model_router = Blueprint('upload_model_router', __name__, static_folder='static',
                                template_folder='templates')

BUCKET_NAME = 'test'
MINIO_URL = '20.2.210.176:9000'
MINIO_ACCESS_KEY = 'minio'
MINIO_SECRET = 'minio123'



@upload_model_router.route('/upload-model-minio', methods=['GET', 'POST'])
def upload_model():
    if request.method == "POST":

        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]

        if file.filename == "":
            return redirect(request.url)
        if file:
            size = os.fstat(file.fileno()).st_size
            file_after_upload = minio_utils.upload_object(file.filename, file, size)

            return redirect(request.url)
    return render_template('admin/upload-model.html')


@upload_model_router.route('/minio', methods=['GET'])
def get_minio_model():
    file_name = request.args.get('file_name')
    # Get data of an object.
    return minio_utils.get_minio_object(file_name)


@upload_model_router.route('/upload-model')
def admin_upload_train_model():
    model_name = request.args.get('model_name')
    model_datas = model.find()

    model_names = [m.get('name') for m in model_datas]

    if (model_name):
        model_data = model.find_one({'name': model_name})
        data = model_data.get('attrs')

        return render_template('admin/upload-model.html',
                               model_names=model_names,
                               data=data,
                               model_name=model_name)

    return render_template('admin/upload-model.html',
                           model_names=model_names,
                           data=None,
                           model_name="")


@upload_model_router.route('/search-upload-model', methods=['GET'])
def get_data_train_model():
    model_name = request.args.get('model_name')
    if not model_name:
        return jsonify("Model name not found")
    model_data_find = model.find_one({'name': model_name})
    data = model_data_find.get('attrs')

    return jsonify(data)


@upload_model_router.route('/upload-model', methods=['POST'])
def admin_train_model():
    try:
        name_train = request.form['name_train']
        model_name = request.args.get('model_name')
        data_name = request.form['data_name']
        data_name = ast.literal_eval(data_name)
        file = request.files["file"]

        if file.filename == "":
            raise Exception("Không tìm thấy file")

        size = os.fstat(file.fileno()).st_size
        file_after_upload = minio_utils.upload_object(file.filename, file, size)

        # kiểm tra model

        factory_model = FactoryModel(model_name).factory()

        model_url = minio_utils.get_minio_object(file_after_upload.object_name)
        if not model_url:
            raise Exception("Upload model thất bại")

        factory_model.model_url = model_url
        factory_model.data_uri = data_name.get('data')

        _, test_data = factory_model.prepare_data_for_self_train()

        n_periods = len(test_data)
        data_predict, acuracy = factory_model.train_for_upload_mode(n_periods, test_data)
        if not data_predict or not acuracy:
            raise Exception("Train model thất bại")

        data_model = {
            "user_id": session.get('username'),
            "name": name_train,
            "model_name": model_name,
            "agricultural_name": data_name['type'],
            "data_name": data_name['data'],
            "file_name": file_after_upload.object_name,
            "file_etag": file_after_upload.etag,
            "create_time": datetime.now(),
            "evaluate": acuracy,
            "type": constant.UPLOAD_MODEL,
            "status": constant.SUCCESS,
            "is_used": False
        }

        train_model.insert_one(data_model)
        return redirect("detail-model?model_id=" + str(data_model.get('_id')))
    except Exception as e:
        flash("Thất bại: " + str(e))
        return redirect("/upload-model")


def create_chart_mode(data_actual, data_predicted, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_actual.index, y=data_actual['price'], mode='lines', name='Giá thực tế',
                             line=dict(color='rgba(0, 0, 255, 0.5)'), fillcolor='rgba(173, 216, 230, 0.2)',
                             visible=True))
    fig.add_trace(go.Scatter(x=data_predicted.index, y=data_predicted['price'], mode='lines',
                             name='Giá dự đoán', line=dict(color='rgba(255, 165, 0, 0.5)'),
                             fillcolor='rgba(255, 165, 0, 0.2)', visible=True))
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
    current_app.logger.info("CREATE CHART " + model_name)
    pio.write_html(fig, './templates/chart/' + model_name + '.html')


@upload_model_router.route('/detail-model')
def admin_detail_model():
    global plot_data
    model_id = request.args.get('model_id')
    current_app.logger.info(model_id)
    model_data = train_model.find_one(ObjectId(model_id))
    current_app.logger.info(model_data)

    if model_data:
        factory_model = FactoryModel(model_data.get('model_name')).factory()
        model_url = minio_utils.get_minio_object(model_data.get('file_name'))
        if not model_url:
            flash("Không tìm thấy model")
            return render_template('admin/detail-model.html',
                           model_data=None,
                           plot_data=None)

        factory_model.model_url = model_url
        factory_model.data_uri = model_data.get('data_name')
        _, test_data = factory_model.prepare_data_for_self_train()

        n_periods = len(test_data)
        data_predict, ac = factory_model.train_for_upload_mode(n_periods, test_data)
        current_app.logger.info(ac)
        if (isinstance(data_predict, pd.DataFrame)):
            data_predict.set_index(test_data.index, inplace=True)
        else:
            data_predict = pd.DataFrame(data_predict, index=test_data.index, columns=['price'])

        data_url = model_data.get('data_name')
        url_parts = data_url.split('/')
        filename = url_parts[-1]
        model_data['data_name'] = filename

        plot_data = []
        trace1 = dict(
            x=test_data.index.tolist(),
            y=test_data.price.values.tolist(),
            mode='lines',
            name='Giá thực tế',
        )
        plot_data.append(trace1)

        trace2 = dict(
            x=data_predict.index.tolist(),
            y=data_predict.price.values.tolist(),
            mode='lines',
            name='Giá dự đoán',
        )
        plot_data.append(trace2)
    else:
        flash('Không tìm thấy mô hình.', 'danger')
    return render_template('admin/detail-model.html',
                           model_data=model_data,
                           plot_data=plot_data)


@upload_model_router.route('/admin')
def admin():
    records = list(train_model.find())
    for record in records:
        data_url = record.get('data_name')
        url_parts = data_url.split('/')
        filename = url_parts[-1]
        record['data_name'] = filename
    return render_template('admin/index.html', train_model_list=records, total_model=len(records))
