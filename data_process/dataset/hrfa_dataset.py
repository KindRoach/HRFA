import torch

from data_process.data_reader.data_enum import DataTypeEnum, DataSetEnum
from data_process.dataset.base_dataset import BaseDataset


class HrfaDataset(BaseDataset):

    def __getitem__(self, index):
        data = self.train_data_file
        tensors = torch.LongTensor(data.user_id[index:index + 1].copy()), \
                  torch.LongTensor(data.user_review[index].copy()), \
 \
                  torch.LongTensor(data.item_id[index:index + 1].copy()), \
                  torch.LongTensor(data.item_review[index].copy()), \
 \
                  torch.LongTensor(data.user_ref_review[index].copy()), \
                  torch.Tensor(data.user_ref_self_rating[index].copy()), \
                  torch.Tensor(data.user_ref_other_rating[index].copy()), \
 \
                  torch.LongTensor(data.item_ref_review[index].copy()), \
                  torch.Tensor(data.item_ref_self_rating[index].copy()), \
                  torch.Tensor(data.item_ref_other_rating[index].copy()), \
 \
                  torch.Tensor(data.rating[index:index + 1].copy())
        return tensors


def test():
    dataset = HrfaDataset(DataSetEnum.Appliances, DataTypeEnum.Train)
    dataset.load_in_order()
    dataset.load_test()


if __name__ == '__main__':
    test()
