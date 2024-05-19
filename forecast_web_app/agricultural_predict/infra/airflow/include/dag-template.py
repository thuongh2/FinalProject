import airflow
from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from pendulum import datetime
from forecast_web_app.agricultural_predict.infra.airflow.dags.test_agg import default_args
from airflow.models.param import Param
from forecast_web_app.agricultural_predict.model.factory_model import FactoryModel

default_args = {
    "params": {
        "model_name": model_name_replace,

        "data_name": data_name_replace,

        "argument": argument_replace,
    }
}

@dag(
    dag_id=dag_id_to_replace,
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    is_paused_upon_creation = False,
    params = {
       
    }
)

def dag_from_config():

    @task(task_id='define_model')
    def define_model():
        factory_model = FactoryModel(params.model_name).factory()
        factory_model.data_uri = params.data_name
        return factory_model

    @task(task_id='load_data')
    def load_data(factory_model):
        factory_model.prepare_data_for_self_train()


    @task(task_id='load_data')
    def train_model(factory_model):
        factory_model.train_model(params.argument)

    model = define_model()
    data = load_data(model)
    train_model(model)


dag_from_config()
