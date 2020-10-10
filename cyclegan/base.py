from abc import ABC, abstractmethod
from typing import NoReturn
from bunch import Bunch
import tensorflow as tf


class BaseModel(ABC):
    def __init__(self, model_config: Bunch):
        super(BaseModel, self).__init__()
        self.model_config = model_config
        self.build_models()

    @abstractmethod
    def build_models(self) -> NoReturn:
        pass

    @abstractmethod
    def train(self, training_data: tf.data.Dataset, validation_data: tf.data.Dataset, epochs: int,
              batch_size: int) -> NoReturn:
        pass
