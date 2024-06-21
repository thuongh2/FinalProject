from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash, jsonify
from flask import session
import hashlib
from flask import current_app
from config.db_conn_config import db, model_info_collection, user_collection, model_registry_collection
import requests
import pandas as pd
import datetime
import warnings
import http
from utils import constant
from cachetools import TTLCache

warnings.simplefilter(action='ignore', category=FutureWarning)

main_router = Blueprint('main_router', __name__, static_folder='static',
                        template_folder='templates')

AGRICULTURAL_TYPES = constant.AGRICULTURAL_TYPES
AGRICULTURAL_TYPES_MAPPING = constant.CONVERT_ARGICULTURAL

cache = TTLCache(maxsize=50, ttl=50000)

@main_router.route('/')
def index():
    records = list(model_info_collection.find())

    price_data = []

    try:
        price_agricultural = cache['price_agricultural']
    except:
        price_agricultural = get_price_with_date()
        cache['price_agricultural'] = price_agricultural

    model_name = request.args.get('model_name', constant.LSTM)
    agricultural_type = request.args.get('agricultural_type', constant.CAFE)

    if (model_name):
        model_data_find = model_info_collection.find_one({'name': model_name})
        data = model_data_find.get('data', [])
        filtered_data = [d for d in data if d.get('type') == agricultural_type]
        
        return render_template('index.html', data=filtered_data,
                               models=records, records_data=None,
                               price_data=price_data, agricultural=agricultural_type,
                               price_agricultural=price_agricultural,
                               agricultural_type=AGRICULTURAL_TYPES,
                               agricultural_mapping=AGRICULTURAL_TYPES_MAPPING)

    return render_template('index.html', models=records,
                           records_data=None, price_data=price_data,
                           data=None, model_name="", price_agricultural=price_agricultural,
                           agricultural_type=AGRICULTURAL_TYPES,
                           agricultural_mapping=AGRICULTURAL_TYPES_MAPPING)


def get_price_with_date():
    # lấy arima làm chuẩn vì arima chỉ có giá
    arima_model = model_info_collection.find_one({'name': 'ARIMA'})
    data = arima_model.get('data', [])
    price_agricultural = {}
    for type in AGRICULTURAL_TYPES:
        for d in data:
            if d.get('type') == type:
                data_csv = d.get('data')
                current_app.logger.info(data_csv)
                df = pd.read_csv(data_csv)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                today = datetime.date.today()
                current_data = df[df['date'].dt.date == today]
                if current_data.empty:
                    current_data = df.iloc[-1]
                current_data['date'] = current_data['date'].strftime('%d-%m-%Y')

                price_agricultural[type] = current_data

    return price_agricultural


@main_router.route('/search-all-models', methods=['GET'])
def get_all_models():
    all_models = []
    models = model_info_collection.find()
    for model_data in models:
        model_info = {
            'name': model_data.get('name'),
        }
        all_models.append(model_info)
    return jsonify(all_models)


@main_router.route('/search-one-model', methods=['GET'])
def get_data_train_model():
    model_name = request.args.get('model_name')

    data_name = request.args.get('data_name')
    model_data_find = model_info_collection.find_one({'name': model_name})
    data = model_data_find.get('attrs')

    return jsonify(data)

@main_router.route('/get-data-from-model/<model_name>', methods=['GET'])
def get_data_from_model(model_name):
    agricultural_type = request.args.get('agricultural_type')
    model_info = model_info_collection.find_one({'name': model_name})
    if model_info:
        data_agricultural = model_info.get('data')
        current_app.logger.info(agricultural_type)
        if agricultural_type:
            data_agricultural = [entry for entry in data_agricultural if entry['type'].startswith(agricultural_type)]
  
        return jsonify(data_agricultural), http.HTTPStatus.OK
    else:
        return jsonify('Model data not found'), http.HTTPStatus.NOT_FOUND


@main_router.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Check if the password and confirm password match
        if password != confirm_password:
            flash('Mật khẩu không khớp. Vui lòng thử lại.', 'danger')
            return redirect('register')

        password = hashlib.md5(password.encode()).hexdigest()

        # Check if the username already exists
        if user_collection.find_one({'username': username}):
            flash('Username already exists. Choose a different one.', 'danger')
        else:
            user_collection.insert_one({'username': username, 'password': password, 'role': 'USER'})
            flash('Registration successful. You can now log in.', 'success')
            return redirect('login')

    return render_template('register.html')


@main_router.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        password = hashlib.md5(password.encode()).hexdigest()
        # Check if the username and password match
        user_login = user_collection.find_one({'username': username, 'password': password})
        if user_login:
            session['username'] = username
            session['is_authen'] = True
            session['role'] = user_login.get('role')

            # Add any additional logic, such as session management
            return redirect('admin')
        else:
            flash('Tên đăng nhập hoặc mật khẩu không đúng. Vui lòng nhập lại', 'danger')

    return render_template('login.html')


@main_router.route('/logout', methods=['GET'])
def logout():
    if session.get('is_authen'):
        session['is_authen'] = False
        del session['username']
    return redirect('/')


@main_router.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
