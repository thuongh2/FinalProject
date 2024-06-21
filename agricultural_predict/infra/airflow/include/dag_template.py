import airflow
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from pendulum import datetime
from airflow.models.param import Param

from model.factory_model import FactoryModel
from airflow.operators.python import get_current_context
from airflow.operators.python import PythonOperator
from steps.preprocess_step import PreprocessStep
from steps.train_step import TrainStep
from steps.stationary_step import StationaryStep
from steps.mlflow_step import MLFlowStep
from airflow import DAG

default_args = {
    "params": {
        "model_name": "{model_name}",

        "data_name": "{data_name}",

        "argument": eval("{argument}"),

        "type": "{type}",

        "agricultural_name": "{agricultural_name}",

        "owner": "{user_name}",

        "model_id": "{model_id}"
    }
}

model = FactoryModel("{model_name}").factory()
preprocess_step = PreprocessStep(model)
stationary_step = StationaryStep()
train_step = TrainStep()
mlflow_step = MLFlowStep()

with DAG(
        dag_id="{dag_id_to_replace}",
        schedule_interval=None,
        catchup=False,
        default_args=default_args,
        is_paused_upon_creation=False, 
) as dag:

    preprocessing_task = PythonOperator(
        task_id="prepare_data",
        python_callable=preprocess_step,
        op_kwargs={
            "data_path": "{{ params.data_name }}",
            "size": 0.8,
        },
    )

    processing_data = PythonOperator(
        task_id="processing_data",
        python_callable=stationary_step,
        op_kwargs={
            "prepare": preprocess_step,
            "model": preprocess_step.model,
            "type": "{{ params.type }}",
        },
    )

    training_task = PythonOperator(
        task_id="training",
        python_callable=train_step,
        provide_context=True,
        op_kwargs={
            "model": preprocess_step.model,
            "data": "{{ params.data_name }}",
            "argument": "{{ params.argument }}",
            "model_name": "{{ params.model_name }}"
        },
    )


    submit_model = PythonOperator(
        task_id="submit_model",
        python_callable=mlflow_step,
        provide_context=True,
        op_kwargs={
            "agricultural_name": "{{ params.agricultural_name }}",
            "user_name": "{{ params.owner }}",
            "model_id": "{{ params.model_id }}",
            "argument": "{{ params.argument }}",
        },
    )

    preprocessing_task >> processing_data >> training_task >> submit_model
