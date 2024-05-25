import requests
import json


class MLFlowStep:

    def __init__(self) -> None:
        self.model = None

    def __call__(self, ti, agricultural_name, user_name, model_id) -> None:
        file_name = ti.xcom_pull(task_ids='training', key="file_name")
        data_url = ti.xcom_pull(task_ids='training', key="data")
        accuracy = ti.xcom_pull(task_ids='training', key="accuracy")
        mlflow_param = ti.xcom_pull(task_ids='training', key="mlflow_param")
        model_name = ti.xcom_pull(task_ids='training', key="model_name")
        params = {
            "file_name": file_name,
            "data_url": data_url,
            "accuracy": json.dumps(accuracy),
            "model_name": model_name,
            "agricultural_name": agricultural_name,
            "user_name": user_name,
            "model_id": model_id
        }
        print(params)
        url = f"https://g79c2ds0-5000.asse.devtunnels.ms/submit_train_model_airflow"
        for retry in range(3):
            response = requests.get(
                url=url,
                params=params
            )
            print(response.__dict__)
            if response.status_code == 200:
                break
            print(f'Retry call api {url} time ', retry)
