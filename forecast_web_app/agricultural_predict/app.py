from flask import Flask, render_template, request, redirect, url_for, flash
from flask import render_template
from flask import send_from_directory
from pymongo import MongoClient
from config.db import db, model, user
from flask import session
from router import main_router
from upload_model_router import upload_model_router

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates')

app.secret_key = "cqH3HoQ1Cp4zafXn"

is_authen = False

app.register_blueprint(main_router)
app.register_blueprint(upload_model_router)


if __name__ == '__main__':
    db.init_app()
    app.run(debug=True)