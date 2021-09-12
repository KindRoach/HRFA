from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from data_process.data_reader.data_enum import DataSetEnum, DataTypeEnum


@dataclass
class BaseConfig(object):
    data_set: DataSetEnum = DataSetEnum.Luxury_Beauty
    steps_num: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    learning_rate_decay: float = 1
    l2_regularization: float = 0
    dropout: float = 0
    device: str = "cpu"

    word_dim: int = 50
    word_count: int = 5000
    user_count: int = 5000
    item_count: int = 5000
    review_count: int = 10
    review_length: int = 1000
    avg_rating: float = 0

    @property
    def pad_id(self):
        return 0


class BaseModel(torch.nn.Module):
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.current_training_step = 0
        self.config = config
        self.loss_f = None
        self.dataset_class = None
        self.mid_var_map = dict()
        self.record_mid_var = False

    def get_name(self):
        return f"{self.__class__.__name__}_{self.config.data_set}"

    def get_device(self) -> torch.device:
        return list(self.parameters())[0].device

    def get_param_number(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def move_iter_i_to_device(self, iter_i):
        return [tensor.to(self.get_device()) for tensor in iter_i]

    def predict_iter_i(self, iter_i):
        iter_i = self.move_iter_i_to_device(iter_i)
        args = iter_i[:-1]
        rating = iter_i[-1]
        predict = self(*args)
        loss = self.loss_f(predict, rating)
        return predict, rating, loss

    def create_dataset(self, data_type: DataTypeEnum) -> Dataset:
        return self.dataset_class(self.config.data_set, data_type)
