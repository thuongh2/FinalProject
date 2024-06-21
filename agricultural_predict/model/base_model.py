from abc import abstractmethod

import numpy as np
import pandas as pd
from pandas import DataFrame
from pyparsing import Optional
from sklearn.metrics import mean_squared_error
from typing import Optional

from utils.minio_utils import get_minio_object
from sklearn.metrics import mean_absolute_percentage_error

class BaseModel:

    def __init__(self) -> None:
        """Initializes the model and data attributes.

        This constructor initializes the following attributes:
          - model: The forecasting model (initially None).
          - model_url: The URL to load the model from (empty string).
          - data_uri: The URL to load the data from (empty string).
          - data: The original loaded data (empty DataFrame).
          - train_data: The training data (empty DataFrame).
          - test_data: The testing data (empty DataFrame).
          - forecast_data: The DataFrame containing forecasts (empty DataFrame).
          - accuracy: A dictionary to store forecast accuracy metrics (empty).
        """

        # Model and data attributes
        self.model: object = None
        self.model_url: str = ""
        self.data_uri: str = ""
        self.data: DataFrame = pd.DataFrame()
        self.train_data: DataFrame = pd.DataFrame()
        self.test_data: DataFrame = pd.DataFrame()
        self.forecast_data: DataFrame = pd.DataFrame()
        self.accuracy: Optional[dict] = None

        self.PRICE_COLUMN: str = 'price'
        self.ML_FLOW_URL: str = 'http://agricultural.io.vn:5000'

    @abstractmethod
    def train_for_upload_mode(self, n_periods, test_data):
        pass

    @abstractmethod
    def train_model(self, argument):
        pass

    @abstractmethod
    def _load_model(self):
        pass
    
    def prepare_data_for_self_train(self, split_size=0.8):
        return self.prepare_data(self.data_uri, split_size)

    def forecast_accuracy(self, actual_value, predicted_values):

        mape = mean_absolute_percentage_error(actual_value, predicted_values) * 100

        mse = mean_squared_error(actual_value, predicted_values)
        rmse = np.sqrt(mse)
        print("rmse: ", rmse)
        print("mape: ", mape)
        return {'mape': round(mape, 2), 'rmse': round(rmse, 2)}

    def prepare_data(self, data_url, split_size=0.8):
        """Reads data from a URL, performs basic cleaning, and splits it into train and test sets.

        Args:
            data_url: The URL of the CSV data to be loaded.
            split_size: The proportion of the data to be used for training (default: 0.8).

        Returns:
            A tuple containing the training and testing DataFrames.

        Raises:
            Exception: If the data at the provided URL is empty.
        """

        print(f'Reading data from {data_url}')
        if 'githubusercontent' not in data_url:
            print('start read data from minio')
            minio_file = data_url.split('/')[-1]
            data_url = get_minio_object(minio_file, 'data')
        self.data = pd.read_csv(data_url)
        print(self.data)
        if self.data.empty:
            raise Exception(f'Data at {data_url} is empty')

        self._clean_data()  # Delegate data cleaning to a separate function

        self.train_data, self.test_data = self._split_data(split_size)  # Delegate splitting to a separate function

        return self.train_data, self.test_data

    def _clean_data(self):
        """Performs basic cleaning on the loaded data (e.g., setting date as index)."""
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.set_index('date', inplace=True)

    def _split_data(self, split_size):
        """Splits the cleaned data into training and testing sets."""
        size = int(len(self.data) * split_size)
        train_data = self.data[:size]
        test_data = self.data[size:]
        return train_data, test_data

    @abstractmethod
    def ml_flow_register(self, experient_name="DEFAUT_MODEL"):
        pass

    def smoothing_data(self, type='exponential', smoothing_value=30):
        if type not in ['exponential', 'moving_average', 'double_exponential']:
            raise ValueError(f"Unknown smoothing type '{type}'.")
        
        if type == 'moving_average':
            smoothing_value = int(smoothing_value) if smoothing_value else 30
            self.data = self.data.rolling(window=smoothing_value, min_periods=1).mean()
        else:
            alpha = float(smoothing_value) if smoothing_value else 0.5
            self.data = self.data.ewm(alpha=alpha, adjust=False).mean()
            if type == 'double_exponential':
                self.data = self.data.ewm(alpha=alpha, adjust=False).mean()

        self.data = self.data.dropna()


