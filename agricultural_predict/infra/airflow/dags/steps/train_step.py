import joblib
import uuid
from minio import Minio

class TrainStep:

    def __init__(self) -> None:
        self.model = None
        self.data_diff = None
        self.predict_data = None
        self.metrics = None

    def __call__(self, ti, model, data, argument, model_name) -> None:
        print(model.__dict__)
        if type(argument) == str:
            argument = eval(argument)
        data = ti.xcom_pull(task_ids='prepare_data', key="data_path")
        self.model = model
        self.model.data_uri = data
        self.predict_data, self.metrics, model_res = self.model.train_model(argument)
        if model_name in ['LSTM', 'GRU', 'BiLSTM']:
            file_name = str(uuid.uuid4()) + '.h5'
            file_dir = "./" + file_name
            model_res.save(file_dir)
        else:
            file_name = str(uuid.uuid4()) + '.joblib'
            file_dir = "./" + file_name
            joblib.dump(model_res, file_dir)

        self.fupload_object(file_name,  file_dir)
        ti.xcom_push(key='file_name', value=file_name)
        ti.xcom_push(key='data', value=data)
        ti.xcom_push(key='accuracy', value=self.metrics)
        ti.xcom_push(key='mlflow_param', value=self.model.ml_flow_param())
        ti.xcom_push(key='model_name', value=model_name)

        return self.metrics


    def fupload_object(self, filename, data, length= None):
        BUCKET_NAME = 'final-project'
        MINIO_URL = 'agricultural.io.vn:9000'
        MINIO_ACCESS_KEY = 'minio'
        MINIO_SECRET = 'minio123'
        PATH = "./file/"
        client = Minio(MINIO_URL, MINIO_ACCESS_KEY, MINIO_SECRET, secure=False)

        # Make bucket if not exist.
        found = client.bucket_exists(BUCKET_NAME)
        if not found:
            client.make_bucket(BUCKET_NAME)
        else:
            print(f"Bucket {BUCKET_NAME} already exists")

        file = client.fput_object(BUCKET_NAME, filename, data)
        print(f"{filename} is successfully uploaded to bucket {BUCKET_NAME}.")
        return file
