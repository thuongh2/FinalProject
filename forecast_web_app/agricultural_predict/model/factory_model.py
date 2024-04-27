from model.arima_model import ARIMAModel
from model.arimax_model import ARIMAXModel
from forecast_web_app.agricultural_predict.model.var_model import VARModel
from forecast_web_app.agricultural_predict.model.varma_model import VARMAModel


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