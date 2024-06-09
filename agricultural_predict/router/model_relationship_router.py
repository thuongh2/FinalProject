import http

from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash, jsonify
from flask import current_app
from config.db_conn_config import model_info_collection, model_data_relations_collection, model_registry_collection
from flask import session

from flask import current_app
from cachetools import TTLCache

import pandas as pd
from werkzeug.utils import secure_filename
from statsmodels.tsa.stattools import acf, pacf

import warnings
from utils import constant
from utils import common_utils

warnings.simplefilter(action='ignore', category=FutureWarning)
cache = TTLCache(maxsize=50, ttl=5000)

model_relationship_router = Blueprint('model_relationship_router',
                            __name__,
                            static_folder='static',
                            template_folder='templates')


@model_relationship_router.route('/model-relationship', methods=['GET', 'POST'])
def create_model_relationship():
    if request.method == 'POST':
        model_name = request.form['model_name']
        model_data = request.form['model_data']
        model_data_name = request.form['model_data_name']
        train_model_id = request.form['train_model_id']
        train_model_name = request.form['train_model_name']

        model_valid = model_data_relations_collection.find_one(
            {'model_name': model_name, 'model_data': model_data}
        
        )
        if (model_valid):
            flash('Mô hình đã tồn tại')
        else:
            model_relationship = {
                '_id': common_utils.generate_id(model_name),
                'model_name': model_name,
                'model_data': model_data,
                'model_data_name': model_data_name,
                'train_model_id': train_model_id,
                'train_model_name': train_model_name,
            }
            model_data_relations_collection.insert_one(model_relationship)
            flash('Thêm mô hình thành công')

    model_relationships = model_data_relations_collection.find()
    try:
        model_info = cache['model_info']
    except:
        model_info = model_info_collection.find()
        cache['model_info'] = model_info

    return render_template('admin/model-relationship.html',
                        model_info=model_info, 
                        model_relationships=model_relationships)


@model_relationship_router.route('/delete-relation-ship/<model_id>', methods=['GET'])
def delete_model_relationship(model_id):
    model_data_relations_collection.delete_one({'_id': model_id})
    flash('Xóa mô hình thành công')
    return redirect('/model-relationship')


@model_relationship_router.route('/get-traning-from-model/<model_name>', methods=['GET'])
def get_traning_from_model(model_name):
    model_data = request.args.get('model_data')
    model_type = request.args.get('model_type')

    query = {
        'model_name': model_name,
        'agricultural_name': model_type,
    }
    traning_model = list(model_registry_collection.find(query))

    if traning_model:
        return jsonify(traning_model), http.HTTPStatus.OK
    else:
        current_app.logger.error('Model traning not found')
        return jsonify('Model data not found'), http.HTTPStatus.NOT_FOUND