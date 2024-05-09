from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash, jsonify
from flask import send_from_directory
from pymongo import MongoClient
from flask import session
import hashlib
from flask import current_app
from config.db import db, model, user, train_model
import requests
import pandas as pd

main_router = Blueprint('main_router', __name__, static_folder='static',
            template_folder='templates')

@main_router.route('/')
def index():
    records = list(model.find())
    records_data = list(train_model.find())
    price_data = []
    seen_algricutural_names = set()
    
    for record_data in records_data:
        data_url = record_data.get('data_name')
        if record_data.get('algricutural_name') not in seen_algricutural_names:
            response = requests.get(data_url)
            
            if response.status_code == 200:
                df = pd.read_csv(data_url)
                last_rows = df.tail(20)
                for index, row in last_rows.iterrows():
                    price_data.insert(0, {
                        'date': row['date'],
                        'price': row['price'],
                        'algricutural_name': record_data.get('algricutural_name')
                    })
                seen_algricutural_names.add(record_data.get('algricutural_name'))
    return render_template('index.html', models=records, records_data=records_data, price_data=price_data)

@main_router.route('/search-model', methods=['GET'])
def get_all_models():
    all_models = []
    models = model.find()
    for model_data in models:
        model_info = {
            'name': model_data.get('name'),
            'attrs': model_data.get('attrs')
        }
        all_models.append(model_info)
    return jsonify(all_models)

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
        
        users_collection = user
        # Check if the username already exists
        if users_collection.find_one({'username': username}):
            flash('Username already exists. Choose a different one.', 'danger')
        else:
            users_collection.insert_one({'username': username, 'password': password, 'role': 'USER'})
            flash('Registration successful. You can now log in.', 'success')
            return redirect('login')

    return render_template('register.html')

@main_router.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        password = hashlib.md5(password.encode()).hexdigest()
        users_collection = user
        # Check if the username and password match
        user_login = users_collection.find_one({'username': username, 'password': password})
        if user_login:
            session['username'] = username
            session['is_authen'] = True
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