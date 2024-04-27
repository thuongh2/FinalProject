from model.arima_model import ARIMAModel
from model.arimax_model import ARIMAXModel
from model.var_model import VARModel
from model.varma_model import VARMAModel


class FactoryModel:

    def __init__(self, model):
        self.model = model

    def factory(self):
        """Factory Method"""
        localizers = {
            "ARIMA": ARIMAModel,
            "ARIMAX": ARIMAXModel,
            "VAR": VARModel,
            "VARMAX": VARMAModel,
        }
        print(self.model)
        return localizers[self.model]()