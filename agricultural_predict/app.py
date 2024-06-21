from flask import Flask
from config.db_conn_config import db
from router.router import main_router
from router.upload_model_router import upload_model_router
from router.load_chart_router import load_chart_router
from router.train_model_router import train_model_router
from router.train_model_rnn_router import train_model_rnn_router
from router.pipeline_router import pipeline_router
from router.model_relationship_router import model_relationship_router
from flask_cors import CORS
from flask_assets import Environment
import logging

app = Flask(__name__,
            static_url_path='',
            static_folder='static',
            template_folder='templates')

assets = Environment(app)
app.logger.setLevel(logging.INFO)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

app.secret_key = "cqH3HoQ1Cp4zafXn"

is_authen = False

app.register_blueprint(main_router)
app.register_blueprint(upload_model_router)
app.register_blueprint(load_chart_router)
app.register_blueprint(train_model_router)
app.register_blueprint(train_model_rnn_router)
app.register_blueprint(pipeline_router)
app.register_blueprint(model_relationship_router)

if __name__ == '__main__':
    db.init_app()
    app.run()
