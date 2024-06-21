import http

from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash, jsonify
from flask import current_app
from flask_cors import cross_origin
from config.db_conn_config import model_registry_collection, model_data_relations_collection
from flask import session
import cachetools
from flask import current_app
from cachetools import TTLCache, LRUCache

import pandas as pd

from model.factory_model import FactoryModel

import os
import datetime
import warnings
from utils import minio_utils
warnings.simplefilter(action='ignore', category=FutureWarning)

load_chart_router = Blueprint('load_chart_router',
                            __name__, 
                            static_folder='static',
                            template_folder='templates')

cache = TTLCache(maxsize=100, ttl=50000)
model_file_path = './file_model/'

# Cache configuration
lru_cache = LRUCache(maxsize=100)

@load_chart_router.route('/load-chart', methods=['GET'])
@cross_origin()
def load_chart():
    model_name = request.args.get('model_name')
    model_data = request.args.get('model_data')
    model_time = request.args.get('model_time')
    model_time = int(model_time)
    n_steps = 15
    current_app.logger.info(model_data)

    file_name, _ = os.path.splitext(os.path.basename(model_data))
    
    model_relation = get_model_relation(model_name=model_name, model_data=model_data)

    if not model_relation:
        current_app.logger.error('not found model relationship')
        model_file_name = get_model_file_local(model_name, file_name)
    else:
        model_train = model_registry_collection.find_one({'_id': model_relation.get('_id')})
        if model_train:
            try:
                minio_file_name = model_train.get(file_name)
                current_app.logger.info('get model from minio' + minio_file_name)
                model_file_name = minio_utils.get_minio_object(minio_file_name, path=model_file_path)
            except:
                current_app.logger.error('fail to get model from minio')
                model_file_name = get_model_file_local(model_name, file_name)
        else:
            current_app.logger.error('not found model relationship')
            model_file_name = get_model_file_local(model_name, file_name)

    current_app.logger.info('get model from file ' + model_file_name)
    model_url = model_file_name
    model = FactoryModel(model_name).factory()
    model_url = model_url
    model.data_uri = model_data
    model.model_url = model_url
    _, test_data = model.prepare_data_for_self_train()

    try:
        predict_data = cache[model_file_name]
    except:
        predict_data = model.forecast_future(model_time, test_data, n_steps)
        cache[model_file_name] = predict_data

    predict_data['date'] = pd.to_datetime(predict_data['date'])
    predict_data.set_index(['date'], inplace=True)
    test_data = test_data.iloc[-90:]
    first_predict_data_row = predict_data.iloc[[0]]
    first_predict_data_row.index = pd.to_datetime(first_predict_data_row.index)
    test_data = pd.concat([test_data, first_predict_data_row])

    plot_data = []
    trace1 = dict(
        x=test_data.index.tolist(),
        y=test_data.price.values.tolist(),
        mode='lines',
        name='Giá thực tế',
    )
    plot_data.append(trace1)

    trace2 = dict(
        x=predict_data.index.tolist(),
        y=predict_data.price.values.tolist(),
        mode='lines',
        name='Giá dự đoán',
    )
    plot_data.append(trace2)

    # get price actual and predict by date
    today = datetime.date.today()
    test_data['date'] = test_data.index
    predict_data['date'] = predict_data.index
    price_actual = test_data[test_data.index.date == today]
    price_forecast = predict_data[predict_data.index.date == today]
    if price_actual.empty:
        # nếu ko có giá hôm này lấy giá cuối của dữ liệu (giá mới nhất)
        price_actual = test_data.iloc[-1]
    if price_forecast.empty:
        # nếu ko có giá hôm này lấy giá đầu của dữ liệu (giá dự đoán cho ngày hôm sau)
        price_forecast = predict_data.iloc[1]

    if type(price_forecast) == pd.DataFrame:
        price_forecast = price_forecast.iloc[0]

    response_data = {'plot_data': plot_data,
                     'price_actual': price_actual.to_json(),
                     'price_forecast': price_forecast.to_json()}
    return jsonify(response_data), http.HTTPStatus.OK

def get_model_from_file(model_file_name):
    try:
        model_filename_dir = cache['model_filename_dir']
    except:
        model_filename_dir = os.listdir(model_file_path)
        cache['model_file_dir'] = model_filename_dir

    matching_filename = next((filename for filename in model_filename_dir if filename.lower().startswith(model_file_name.lower())), None)
    return matching_filename

def get_model_file_local(model_name, file_name):
    # not get find from minio and get from local
    model_name_file_path = f'{model_name}_{file_name}'
    current_app.logger.info('get model from local ' + model_name_file_path)
    return model_file_path +  get_model_from_file(model_name_file_path)

@cachetools.cached(cache, key=lambda model_name, model_data: (model_name, model_data))
def get_model_relation(model_name, model_data):
    model_relation = model_data_relations_collection.find_one({
        'model_name': model_name,
        'model_data': model_data,
    })
    return model_relation