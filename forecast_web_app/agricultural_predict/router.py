from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash
from flask import render_template
from flask import send_from_directory
from pymongo import MongoClient
from config.db import db, model, user
from flask import session
import hashlib
from flask import current_app


main_router = Blueprint('main_router', __name__, static_folder='static',
            template_folder='templates')

@main_router.route('/')
def hello():
    return render_template('index.html')


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


@main_router.route('/admin')
def admin():
    return render_template('admin/index.html')

@main_router.route('/upload-model')
def admin_upload_train_model():
    return render_template('admin/upload-model.html')

@main_router.route('/detail-model')
def admin_detail_model():
    return render_template('admin/detail-model.html')

@main_router.route('/train-model')
def admin_train_model():
    return render_template('admin/train-model.html')



@main_router.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404