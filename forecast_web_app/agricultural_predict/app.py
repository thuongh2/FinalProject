from flask import Flask, render_template, request, redirect, url_for, flash
from flask import render_template
from flask import send_from_directory
from pymongo import MongoClient
from config.db import db, model, user
from flask import session

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates')

app.secret_key = "cqH3HoQ1Cp4zafXn"

is_authen = False

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        users_collection = user
        # Check if the username already exists
        if users_collection.find_one({'username': username}):
            flash('Username already exists. Choose a different one.', 'danger')
        else:
            users_collection.insert_one({'username': username, 'password': password, 'role': 'USER'})
            flash('Registration successful. You can now log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        users_collection = user
        # Check if the username and password match
        user_login = users_collection.find_one({'username': username, 'password': password})
        if user_login:
            session['username'] = username
            session['is_authen'] = True
            # Add any additional logic, such as session management
            return redirect(url_for('admin'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')

    return render_template('login.html')


@app.route('/admin')
def admin():
    return render_template('admin/index.html')

@app.route('/hello/')
@app.route('/hello/<name>')
def hello_name(name=None):
    return render_template('hello.html', name=name)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    db.init_app()
    admin.init_app(app)
    app.run(debug=True)