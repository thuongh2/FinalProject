from steps.minio_step import get_minio_object

class PreprocessStep:
    def __init__(self, model) -> None:
        self.model = model

    def __call__(self,ti, data_path: str, size = 0.8, is_auto = False) -> None:
        if is_auto:
            data_path = get_minio_object(data_path)

        print(data_path)
        self.model.data_uri = data_path
        self.model.prepare_data_for_self_train(size)
        ti.xcom_push(key='data_path', value=data_path)
        return data_path

