import torch

from data_process.data_reader.data_enum import DataTypeEnum, DataSetEnum
from data_process.dataset.base_dataset import BaseDataset


class MrmrpDataset(BaseDataset):

    def __getitem__(self, index):
        data = self.train_data_file
        tensors = torch.LongTensor(data.user_review_in_one[index].copy()), \
                  torch.LongTensor(data.item_review_in_one[index].copy()), \
                  torch.LongTensor(data.user_supplementary_review_in_one[index].copy()), \
                  torch.LongTensor(data.user_id[index:index + 1].copy()), \
                  torch.LongTensor(data.item_id[index:index + 1].copy()), \
                  torch.Tensor(data.rating[index:index + 1].copy())
        return tensors


def test():
    dataset = MrmrpDataset(DataSetEnum.Appliances, DataTypeEnum.Train)
    dataset.load_in_order()
    dataset.load_test()


if __name__ == '__main__':
    test()
