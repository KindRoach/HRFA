from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from data_process.data_reader.data_enum import DataSetEnum, DataTypeEnum, REVIEW_COUNT, REVIEW_LENGTH
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR, create_folder

TRAIN_DATA_PATH = "out/train_data/%s/%s"


class TrainDataFile:

    def __init__(self, data_set: DataSetEnum, data_type: DataTypeEnum):
        self.data_len = 0
        self.data_set = data_set
        self.data_type = data_type
        self.out_path = ROOT_DIR.joinpath(TRAIN_DATA_PATH % (self.data_set, self.data_type))

        self.user_id = self.out_path.joinpath("user_id.npy")
        self.item_id = self.out_path.joinpath("item_id.npy")
        self.rating = self.out_path.joinpath("rating.npy")

        # For DeepCoNN
        self.user_review_in_one = self.out_path.joinpath("user_review_in_one.npy")
        self.item_review_in_one = self.out_path.joinpath("item_review_in_one.npy")

        # For PARL
        self.user_auxiliary_review_in_one = self.out_path.joinpath("user_auxiliary_review_in_one.npy")

        # For Mrmrp
        self.user_supplementary_review_in_one = self.out_path.joinpath("user_supplementary_review_in_one.npy")

        # For PARL & MRMRP ablation
        self.user_ref_review_in_one = self.out_path.joinpath("user_ref_review_in_one.npy")

        # For NARRE
        self.user_review = self.out_path.joinpath("user_review.npy")
        self.reviewed_item = self.out_path.joinpath("reviewed_item.npy")
        self.item_review = self.out_path.joinpath("item_review.npy")
        self.reviewing_user = self.out_path.joinpath("reviewing_user.npy")

        # For ablation exp (HRFA-SIM)
        self.user_pos_review = self.out_path.joinpath("user_pos_review.npy")
        self.user_pos_user_id = self.out_path.joinpath("user_pos_user_id.npy")
        self.user_pos_item_id = self.out_path.joinpath("user_pos_item_id.npy")
        self.item_pos_review = self.out_path.joinpath("item_pos_review.npy")
        self.item_pos_user_id = self.out_path.joinpath("item_pos_user_id.npy")
        self.item_pos_item_id = self.out_path.joinpath("item_pos_item_id.npy")

        self.user_ref_review = self.out_path.joinpath("user_ref_review.npy")
        self.user_ref_user_id = self.out_path.joinpath("user_ref_user_id.npy")
        self.user_ref_item_id = self.out_path.joinpath("user_ref_item_id.npy")
        self.user_ref_self_rating = self.out_path.joinpath("user_ref_self_rating.npy")
        self.user_ref_other_rating = self.out_path.joinpath("user_ref_other_rating.npy")

        self.item_ref_review = self.out_path.joinpath("item_ref_review.npy")
        self.item_ref_user_id = self.out_path.joinpath("item_ref_user_id.npy")
        self.item_ref_item_id = self.out_path.joinpath("item_ref_item_id.npy")
        self.item_ref_self_rating = self.out_path.joinpath("item_ref_self_rating.npy")
        self.item_ref_other_rating = self.out_path.joinpath("item_ref_other_rating.npy")

    def read_file(self, file_mode="r"):
        if not isinstance(self.user_id, Path):
            logger.warning("Train data file already created or read.")
            return

        self.user_id = np.load(self.user_id, file_mode)
        self.item_id = np.load(self.item_id, file_mode)
        self.rating = np.load(self.rating, file_mode)

        self.user_review_in_one = np.load(self.user_review_in_one, file_mode)
        self.item_review_in_one = np.load(self.item_review_in_one, file_mode)

        self.user_auxiliary_review_in_one = np.load(self.user_auxiliary_review_in_one, file_mode)
        self.user_supplementary_review_in_one = np.load(self.user_supplementary_review_in_one, file_mode)
        self.user_ref_review_in_one = np.load(self.user_ref_review_in_one, file_mode)

        self.user_review = np.load(self.user_review, file_mode)
        self.reviewed_item = np.load(self.reviewed_item, file_mode)
        self.item_review = np.load(self.item_review, file_mode)
        self.reviewing_user = np.load(self.reviewing_user, file_mode)

        self.user_pos_review = np.load(self.user_pos_review, file_mode)
        self.user_pos_user_id = np.load(self.user_pos_user_id, file_mode)
        self.user_pos_item_id = np.load(self.user_pos_item_id, file_mode)
        self.item_pos_review = np.load(self.item_pos_review, file_mode)
        self.item_pos_user_id = np.load(self.item_pos_user_id, file_mode)
        self.item_pos_item_id = np.load(self.item_pos_item_id, file_mode)

        self.user_ref_review = np.load(self.user_ref_review, file_mode)
        self.user_ref_user_id = np.load(self.user_ref_user_id, file_mode)
        self.user_ref_item_id = np.load(self.user_ref_item_id, file_mode)
        self.user_ref_self_rating = np.load(self.user_ref_self_rating, file_mode)
        self.user_ref_other_rating = np.load(self.user_ref_other_rating, file_mode)

        self.item_ref_review = np.load(self.item_ref_review, file_mode)
        self.item_ref_user_id = np.load(self.item_ref_user_id, file_mode)
        self.item_ref_item_id = np.load(self.item_ref_item_id, file_mode)
        self.item_ref_self_rating = np.load(self.item_ref_self_rating, file_mode)
        self.item_ref_other_rating = np.load(self.item_ref_other_rating, file_mode)

        self.data_len = self.user_id.shape[0]

    def create_file(self, data_len: int):
        if not isinstance(self.user_id, Path):
            logger.warning("Train data file already created or read.")
            return

        self.data_len = data_len
        create_folder(self.out_path)

        self.user_id = open_memmap(self.user_id, "w+", dtype=np.int32, shape=(self.data_len,))
        self.item_id = open_memmap(self.item_id, "w+", dtype=np.int32, shape=(self.data_len,))
        self.rating = open_memmap(self.rating, "w+", dtype=np.float32, shape=(self.data_len,))

        self.user_review_in_one = open_memmap(self.user_review_in_one, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT * REVIEW_LENGTH))
        self.item_review_in_one = open_memmap(self.item_review_in_one, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT * REVIEW_LENGTH))
        self.user_auxiliary_review_in_one = open_memmap(self.user_auxiliary_review_in_one, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT * REVIEW_LENGTH))
        self.user_supplementary_review_in_one = open_memmap(self.user_supplementary_review_in_one, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT * REVIEW_LENGTH))
        self.user_ref_review_in_one = open_memmap(self.user_ref_review_in_one, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT * REVIEW_LENGTH))

        self.user_review = open_memmap(self.user_review, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT, REVIEW_LENGTH))
        self.reviewed_item = open_memmap(self.reviewed_item, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT))
        self.item_review = open_memmap(self.item_review, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT, REVIEW_LENGTH))
        self.reviewing_user = open_memmap(self.reviewing_user, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT))

        self.user_pos_review = open_memmap(self.user_pos_review, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT, REVIEW_LENGTH))
        self.user_pos_user_id = open_memmap(self.user_pos_user_id, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT))
        self.user_pos_item_id = open_memmap(self.user_pos_item_id, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT))

        self.item_pos_review = open_memmap(self.item_pos_review, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT, REVIEW_LENGTH))
        self.item_pos_user_id = open_memmap(self.item_pos_user_id, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT))
        self.item_pos_item_id = open_memmap(self.item_pos_item_id, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT))

        self.user_ref_review = open_memmap(self.user_ref_review, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT, REVIEW_LENGTH))
        self.user_ref_user_id = open_memmap(self.user_ref_user_id, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT))
        self.user_ref_item_id = open_memmap(self.user_ref_item_id, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT))
        self.user_ref_self_rating = open_memmap(self.user_ref_self_rating, "w+", dtype=np.float32, shape=(self.data_len, REVIEW_COUNT))
        self.user_ref_other_rating = open_memmap(self.user_ref_other_rating, "w+", dtype=np.float32, shape=(self.data_len, REVIEW_COUNT))

        self.item_ref_review = open_memmap(self.item_ref_review, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT, REVIEW_LENGTH))
        self.item_ref_user_id = open_memmap(self.item_ref_user_id, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT))
        self.item_ref_item_id = open_memmap(self.item_ref_item_id, "w+", dtype=np.int32, shape=(self.data_len, REVIEW_COUNT))
        self.item_ref_self_rating = open_memmap(self.item_ref_self_rating, "w+", dtype=np.float32, shape=(self.data_len, REVIEW_COUNT))
        self.item_ref_other_rating = open_memmap(self.item_ref_other_rating, "w+", dtype=np.float32, shape=(self.data_len, REVIEW_COUNT))

    def flush(self):
        self.user_id.flush()
        self.item_id.flush()
        self.rating.flush()

        self.user_review_in_one.flush()
        self.item_review_in_one.flush()
        self.user_auxiliary_review_in_one.flush()
        self.user_supplementary_review_in_one.flush()
        self.user_ref_review_in_one.flush()

        self.user_review.flush()
        self.reviewed_item.flush()
        self.item_review.flush()
        self.reviewing_user.flush()

        self.user_pos_review.flush()
        self.user_pos_user_id.flush()
        self.user_pos_item_id.flush()
        self.item_pos_review.flush()
        self.item_pos_user_id.flush()
        self.item_pos_item_id.flush()

        self.user_ref_review.flush()
        self.user_ref_user_id.flush()
        self.user_ref_item_id.flush()
        self.user_ref_self_rating.flush()
        self.user_ref_other_rating.flush()

        self.item_ref_review.flush()
        self.item_ref_user_id.flush()
        self.item_ref_item_id.flush()
        self.item_ref_self_rating.flush()
        self.item_ref_other_rating.flush()


def test():
    data_len = 10

    data = TrainDataFile(DataSetEnum.Test_Set, DataTypeEnum.Train)
    data.create_file(data_len)
    data.flush()

    data = TrainDataFile(DataSetEnum.Test_Set, DataTypeEnum.Train)
    data.read_file()
    assert data.item_id.shape[0] == data_len


if __name__ == '__main__':
    test()
