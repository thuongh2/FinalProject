class StationaryStep:
    def __init__(self) -> None:
        self.model = None
        self.data_diff = None

    def __call__(self, prepare, model, type=None, lag=None) -> None:
        self.data_diff = model.difference_dataset(type, lag)
