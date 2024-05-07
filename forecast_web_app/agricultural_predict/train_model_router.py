import http

from flask import Blueprint
from flask import render_template, request, redirect, url_for, flash, jsonify
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


train_model_router = Blueprint('train_model_router', __name__, static_folder='static',
            template_folder='templates')


@train_model_router.route('/train-model', methods=['GET'])
def train_model():
    current_app.logger.info("VO")
    model_name = request.args.get('model_name')
    model_data = model.find()

    model_names = [m.get('name') for m in model_data]
    current_app.logger.info(model_names)
   
    if(model_name):
        model_data_find = model.find_one({'name': model_name})
        data = model_data_find.get('attrs')
    
        return render_template('admin/train-model.html', model_names=model_names, data=data, model_name=model_name
                               )

    return render_template('admin/train-model.html', 
                           model_names=model_names, data=None, model_name="")



@train_model_router.route('/search-train-model', methods=['GET'])
def get_data_train_model():
    model_data = request.args.get('model_data')

    data = pd.read_csv(model_data)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index(['date'], inplace=True)
    if data.empty:
        return jsonify({'error': 'Data is empty'})
    plot_data = []
    for columns in data.columns:
        data[columns] = data[columns].astype(float)
       
        trace = dict(
            x = data.index.tolist(),
            y = data[columns].values.tolist(),
            mode = 'lines'
        )
        plot_data.append(trace)

    return plot_data
