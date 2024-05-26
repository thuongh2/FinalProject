from model.arima_model import ARIMAModel
from model.arimax_model import ARIMAXModel
from model.var_model import VARModel
from model.varma_model import VARMAModel
from model.lstm_model import LSTMModel
from model.gru_model import GRUModel
from model.bilstm_model import BiLSTMModel


class FactoryModel:

    def __init__(self, model):
        self.model = model

    def factory(self):
        """Factory Method"""
        localizers = {
            "ARIMA": ARIMAModel,
            "SARIMA": ARIMAModel,
            "ARIMAX": ARIMAXModel,
            "VAR": VARModel,
            "VARMA": VARMAModel,
            "LSTM": LSTMModel,
            "GRU": GRUModel,
            "BiLSTM": BiLSTMModel,
        }
    
        return localizers[self.model]()