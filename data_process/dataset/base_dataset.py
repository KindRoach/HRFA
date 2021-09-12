import datetime

from torch.utils.data import Dataset
from tqdm import tqdm

from data_process.data_reader.data_enum import DataTypeEnum, get_all_dataset, DataSetEnum
from data_process.data_reader.process_raw_data import get_user_count, get_item_count, load_processed_data, get_avg_rating
from data_process.data_reader.train_data_file import TrainDataFile
from tool.log_helper import logger


class BaseDataset(Dataset):
    def __init__(self, data_set: DataSetEnum, data_type: DataTypeEnum):
        all_data = load_processed_data(data_set)
        self.data_set = data_set
        self.user_count = get_user_count(all_data)
        self.item_count = get_item_count(all_data)
        self.avg_rating = get_avg_rating(all_data)

        self.data_type = data_type
        self.train_data_file = TrainDataFile(data_set, data_type)
        self.train_data_file.read_file()

    def __len__(self):
        return self.train_data_file.data_len

    def __getitem__(self, index):
        raise NotImplementedError

    def load_test(self):
        batches = self.get_data_loader(4096, "cpu")
        before = datetime.datetime.now()

        for tensors in tqdm(batches, desc="Loading training data"):
            pass

        shapes = "\n".join([t.shape.__str__() for t in tensors])
        logger.info(f"\n{shapes}")

        after = datetime.datetime.now()
        logger.info(f"total time usage: {after - before}")

    def load_in_order(self):
        for i in tqdm(range(len(self)), desc="Loading training data"):
            tensors = self[i]


def test():
    for data_set in get_all_dataset():
        for data_type in [
            DataTypeEnum.Train,
            DataTypeEnum.Dev,
            DataTypeEnum.Test
        ]:
            data = BaseDataset(DataSetEnum.Luxury_Beauty, data_type)
            logger.info(f"{data_set}_{data_type} = {len(data)}")


if __name__ == '__main__':
    test()
