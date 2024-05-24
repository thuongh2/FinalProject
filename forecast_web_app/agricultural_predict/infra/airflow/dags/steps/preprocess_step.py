class PreprocessStep:
    def __init__(self, model) -> None:
        self.model = model


    def __call__(self, data_path: str, size = 0.8) -> None:
        print(data_path)
        self.model.data_uri = data_path
        self.model.prepare_data_for_self_train(size)