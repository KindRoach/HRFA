import math
import random
import sqlite3
from typing import List

import numpy
import pandas
from pandas import DataFrame
from tqdm import tqdm

from data_process.data_reader.data_enum import DataSetEnum, DataTypeEnum, REVIEW_COUNT, RANDOM_STATE, get_all_dataset
from data_process.data_reader.process_raw_data import load_processed_data, split_train_dev_test_data
from data_process.data_reader.train_data_file import TrainDataFile, TRAIN_DATA_PATH
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR


class DataUtil:
    def __init__(self, data: DataFrame, known_data: DataFrame,
                 data_set: DataSetEnum, data_type: DataTypeEnum):

        self.data = data
        self.known_data = known_data
        self.data_set = data_set
        self.data_type = data_type

        self.table_name = data_set.value
        self.db_conn = self.create_memory_db()
        self.rand = random.Random(RANDOM_STATE)

        self.review_cache, self.rating_cache = self.get_review_rating_cache()
        self.user_sim_cache = dict()

    def __del__(self):
        # delete in-memory db
        self.db_conn.close()

    def create_memory_db(self):
        logger.info(f"creating in-memory db {self.table_name}...")
        conn = sqlite3.connect(':memory:')
        c = conn.cursor()
        self.known_data[["userID", "itemID", "rating"]].to_sql(self.table_name, conn, index=False)
        c.execute(f"CREATE INDEX user_idx on {self.table_name} (userID)")
        c.execute(f"CREATE INDEX item_idx on {self.table_name} (itemID)")
        c.execute(f"CREATE INDEX user_item_idx on {self.table_name} (userID, itemID)")
        conn.commit()
        return conn

    def get_review_rating_cache(self):
        logger.info(f"creating review and rating cache...")
        review_cache, rating_cache = dict(), dict()
        iterator = zip(self.known_data["userID"], self.known_data["itemID"], self.known_data["review"], self.known_data["rating"])
        for user, item, review, rating in iterator:
            if user in review_cache:
                review_cache[user][item] = review
                rating_cache[user][item] = rating

            else:
                review_cache[user] = {item: review}
                rating_cache[user] = {item: rating}
        return review_cache, rating_cache

    def create_data_file(self):
        data_len = len(self.data)
        train_data = TrainDataFile(self.data_set, self.data_type)
        train_data.create_file(data_len)

        for idx, row in tqdm(enumerate(self.data.itertuples()), total=data_len, desc="Generating Train Data"):
            user, item = row.userID, row.itemID
            train_data.user_id[idx] = user
            train_data.item_id[idx] = item
            train_data.rating[idx] = row.rating
            self.write_review_in_one(train_data, idx, user, item)
            self.write_self_review(train_data, idx, user, item)
            self.write_pos_review(train_data, idx, user, item)
            self.write_ref_review(train_data, idx, user, item)

        train_data.flush()
        logger.info(f"Train file saved to {train_data.out_path}")

    def write_self_review(self, train_data: TrainDataFile, idx, user, item):
        reviews, ids = self.get_user_item_review(user, item, "user_review")
        chunk_len = len(reviews)
        if chunk_len > 0:
            train_data.user_review[idx, 0:chunk_len] = reviews
            train_data.reviewed_item[idx, 0:chunk_len] = ids

        reviews, ids = self.get_user_item_review(item, user, "item_review")
        chunk_len = len(reviews)
        if chunk_len > 0:
            train_data.item_review[idx, 0:chunk_len] = reviews
            train_data.reviewing_user[idx, 0:chunk_len] = ids

    def get_user_item_review(self, query_id: int, exclude_id: int, review_type: str):

        if review_type == "user_review":
            query_key = "userID"
            exclude_key = "itemID"
            user_id_index = 0
            item_id_index = 1
        else:
            query_key = "itemID"
            exclude_key = "userID"
            user_id_index = 1
            item_id_index = 0

        c = self.db_conn.cursor()
        cursor = c.execute(f"select {query_key}, {exclude_key} "
                           f"from {self.table_name} "
                           f"where {query_key} == {query_id} and {exclude_key} != {exclude_id}")

        # Sample REVIEW_COUNT reviews.
        # TODO Better review choice policy
        cursor = self.reservoir_sampling(cursor, REVIEW_COUNT)

        reviews, ids = [], []
        for row in cursor:
            reviews.append(self.review_cache[row[user_id_index]][row[item_id_index]])
            ids.append(row[1])

        return reviews, ids

    def write_review_in_one(self, train_data: TrainDataFile, idx, user, item):
        review, _ = self.get_user_item_review_in_one(user, item, "user_review")
        train_data.user_review_in_one[idx, 0:len(review)] = review

        review, _ = self.get_user_item_review_in_one(item, user, "item_review")
        train_data.item_review_in_one[idx, 0:len(review)] = review

        review = self.get_user_auxiliary_review_in_one(user, item)
        train_data.user_auxiliary_review_in_one[idx, 0:len(review)] = review

        review = self.get_user_supplementary_review_in_one(user, item)
        train_data.user_supplementary_review_in_one[idx, 0:len(review)] = review

        review = self.get_user_ref_review_in_one(user, item)
        train_data.user_ref_review_in_one[idx, 0:len(review)] = review

    def get_user_item_review_in_one(self, query_id: int, exclude_id: int, review_type: str):

        if review_type == "user_review":
            query_key = "userID"
            exclude_key = "itemID"
        else:
            query_key = "itemID"
            exclude_key = "userID"

        c = self.db_conn.cursor()
        cursor = c.execute(f"select userID, itemID "
                           f"from {self.table_name} "
                           f"where {query_key} == {query_id} and {exclude_key} != {exclude_id}")

        reviews = self.read_review_in_one_from_db(cursor, REVIEW_COUNT)
        return reviews

    def get_user_auxiliary_review_in_one(self, user_id: int, exclude_item: int):
        review, still_need = self.get_user_auxiliary_review_in_one_diff(user_id, exclude_item, 0)

        # More auxiliary at rating + 1
        if still_need > 0:
            new_review, still_need = self.get_user_auxiliary_review_in_one_diff(user_id, exclude_item, 1, still_need)
            review.extend(new_review)

        # More auxiliary at rating - 1
        if still_need > 0:
            new_review, _ = self.get_user_auxiliary_review_in_one_diff(user_id, exclude_item, -1, still_need)
            review.extend(new_review)

        return review

    def get_user_auxiliary_review_in_one_diff(self, user_id: int, exclude_item: int, rating_diff: int, limit: int = REVIEW_COUNT):
        """
        Query and read auxiliary review with given condition.
        :param user_id: User id.
        :param exclude_item: Item id.
        :param rating_diff: The rating difference. E.g.: 0 means same ratting.
        :param limit: How many reviews is needed.
        :return: review document, how many reviews is still needed.
        """

        c = self.db_conn.cursor()
        cursor = c.execute(f"select y.userID as refUser, y.itemID as refItem "
                           f"from {self.table_name} as x inner join {self.table_name} as y on x.itemID == y.itemID "
                           f"where x.userID == {user_id} "
                           f"and x.userID != y.userID "
                           f"and y.rating == (x.rating + {rating_diff}) "
                           f"and y.itemID != {exclude_item}")

        reviews, review_count = self.read_review_in_one_from_db(cursor, limit)
        return reviews, limit - review_count

    def read_review_in_one_from_db(self, cursor, limit: int) -> (list, int):
        """
        Read and join multi reviews into one document.
        :param cursor: Database query result.
        :param limit: How many reviews is needed.
        :return: review document, how many reviews read actually.
        """
        # Sample REVIEW_COUNT reviews.
        # TODO Better review choice policy
        cursor = self.reservoir_sampling(cursor, limit)
        reviews = []
        review_read = 0
        for row in cursor:
            review_read += 1
            review_i = self.review_cache[row[0]][row[1]]
            review_i = [word for word in review_i if word != 0]  # Remove <pad>.
            reviews.extend(review_i)
        return reviews, review_read

    def get_user_supplementary_review_in_one(self, user_id: int, exclude_item: int):
        reviews = []
        bought_items = self.rating_cache[user_id].keys()
        for item_id in bought_items:
            if item_id != exclude_item:
                sim_buyers = self.get_similar_buyer(user_id, item_id)
                if len(sim_buyers) == 0:
                    return []

                similarity = [self.get_user_similarity(user_id, sim_buyer) for sim_buyer in sim_buyers]
                max_index = numpy.argmax(similarity).item()
                reviews.append(self.review_cache[sim_buyers[max_index]][item_id])
                if len(reviews) >= REVIEW_COUNT:
                    break

        review_in_one = []
        for r in reviews:
            review_in_one.extend([word for word in r if word != 0])  # Remove <pad>.
        return review_in_one

    def get_similar_buyer(self, user_id: int, item_id: int) -> List[int]:
        self_rating = self.rating_cache[user_id][item_id]
        c = self.db_conn.cursor()
        cursor = c.execute(f"select userID "
                           f"from {self.table_name} "
                           f"where itemID == {item_id} "
                           f"and userId != {user_id} "
                           f"and abs(rating - {self_rating}) <= 1")
        return [row[0] for row in cursor]

    def get_user_similarity(self, user1: int, user2: int) -> float:
        if user1 == user2:
            logger.warning("Calculate similarity between same user.")
            return 1

        key = (min(user1, user2), max(user1, user2))
        if key in self.user_sim_cache:
            return self.user_sim_cache[key]
        else:
            sim = self.cal_user_similarity(user1, user2)
            self.user_sim_cache[key] = sim
            return sim

    def cal_user_similarity(self, user1: int, user2: int) -> float:
        items1 = self.rating_cache[user1].keys()
        items2 = self.rating_cache[user2].keys()
        common_items = set(items1) & set(items2)

        ratings1 = numpy.array([self.rating_cache[user1][item] for item in items1])
        ratings2 = numpy.array([self.rating_cache[user2][item] for item in items2])
        common_ratings1 = numpy.array([self.rating_cache[user1][item] for item in common_items])
        common_ratings2 = numpy.array([self.rating_cache[user2][item] for item in common_items])

        temp = ((common_ratings1 - common_ratings1.mean()) * (common_ratings2 - common_ratings2.mean())).sum()
        sim = math.sqrt(len(common_items)) * temp / (ratings1.std() * ratings2.std())

        if numpy.isnan(sim):
            sim = - numpy.inf
        return sim

    def get_user_ref_review_in_one(self, user_id: int, exclude_item: int):
        c = self.db_conn.cursor()
        cursor = c.execute(f"select y.userID as refUser, y.itemID as refItem "
                           f"from {self.table_name} as x inner join {self.table_name} as y on x.itemID == y.itemID "
                           f"where x.userID == {user_id} "
                           f"and x.userID != y.userID "
                           f"and y.itemID != {exclude_item}")

        reviews, review_count = self.read_review_in_one_from_db(cursor, REVIEW_COUNT)
        return reviews

    def write_pos_review(self, train_data: TrainDataFile, idx, user, item):
        reviews, user_ids, item_ids = self.get_user_item_pos_review(user, item, "user_review")
        chunk_len = len(reviews)
        if chunk_len > 0:
            train_data.user_pos_review[idx, 0:chunk_len] = reviews
            train_data.user_pos_user_id[idx, 0:chunk_len] = user_ids
            train_data.user_pos_item_id[idx, 0:chunk_len] = item_ids

        reviews, user_ids, item_ids = self.get_user_item_pos_review(item, user, "item_review")
        chunk_len = len(reviews)
        if chunk_len > 0:
            train_data.item_pos_review[idx, 0:chunk_len] = reviews
            train_data.item_pos_user_id[idx, 0:chunk_len] = user_ids
            train_data.item_pos_item_id[idx, 0:chunk_len] = item_ids

    def get_user_item_pos_review(self, query_id: int, exclude_id: int, review_type: str):
        if review_type == "user_review":
            query_key = "userID"
            join_key = "itemID"
        else:
            query_key = "itemID"
            join_key = "userID"

        c = self.db_conn.cursor()
        cursor = c.execute(f"select y.userID as refUser, y.itemID as refItem "
                           f"from {self.table_name} as x inner join {self.table_name} as y on x.{join_key} == y.{join_key} "
                           f"where x.{query_key} == {query_id} "
                           f"and x.{query_key} != y.{query_key} "
                           f"and y.{join_key} != {exclude_id} "
                           f"and abs(x.rating - y.rating) <= 1")

        # Sample REVIEW_COUNT reviews.
        # TODO Better review choice policy
        cursor = self.reservoir_sampling(cursor, REVIEW_COUNT)

        reviews, user_ids, item_ids = [], [], []
        for row in cursor:
            reviews.append(self.review_cache[row[0]][row[1]])
            user_ids.append(row[0])
            item_ids.append(row[1])

        return reviews, user_ids, item_ids

    def write_ref_review(self, train_data: TrainDataFile, idx, user, item):
        reviews, user_ids, item_ids, self_rating, other_rating = self.get_user_item_ref_review(user, item, "user_review")
        chunk_len = len(reviews)
        if chunk_len > 0:
            train_data.user_ref_review[idx, 0:chunk_len] = reviews
            train_data.user_ref_user_id[idx, 0:chunk_len] = user_ids
            train_data.user_ref_item_id[idx, 0:chunk_len] = item_ids
            train_data.user_ref_self_rating[idx, 0:chunk_len] = self_rating
            train_data.user_ref_other_rating[idx, 0:chunk_len] = other_rating

        reviews, user_ids, item_ids, self_rating, other_rating = self.get_user_item_ref_review(item, user, "item_review")
        chunk_len = len(reviews)
        if chunk_len > 0:
            train_data.item_ref_review[idx, 0:chunk_len] = reviews
            train_data.item_ref_user_id[idx, 0:chunk_len] = user_ids
            train_data.item_ref_item_id[idx, 0:chunk_len] = item_ids
            train_data.item_ref_self_rating[idx, 0:chunk_len] = self_rating
            train_data.item_ref_other_rating[idx, 0:chunk_len] = other_rating

    def get_user_item_ref_review(self, query_id: int, exclude_id: int, review_type: str):

        if review_type == "user_review":
            query_key = "userID"
            join_key = "itemID"
        else:
            query_key = "itemID"
            join_key = "userID"

        c = self.db_conn.cursor()
        cursor = c.execute(f"select y.userID as refUser, y.itemID as refItem, x.rating as selfRating, y.rating as otherRating "
                           f"from {self.table_name} as x inner join {self.table_name} as y on x.{join_key} == y.{join_key} "
                           f"where x.{query_key} == {query_id} "
                           f"and x.{query_key} != y.{query_key} "
                           f"and y.{join_key} != {exclude_id}")

        # Sample REVIEW_COUNT reviews.
        # TODO Better review choice policy
        cursor = self.reservoir_sampling(cursor, REVIEW_COUNT)

        reviews, user_ids, item_ids, self_rating, other_rating = [], [], [], [], []
        for row in cursor:
            reviews.append(self.review_cache[row[0]][row[1]])
            user_ids.append(row[0])
            item_ids.append(row[1])
            self_rating.append(row[2])
            other_rating.append(row[3])

        return reviews, user_ids, item_ids, self_rating, other_rating

    def reservoir_sampling(self, cursor, sample_k: int):
        """
        Memory friendly sample method.
        https://www.geeksforgeeks.org/reservoir-sampling/
        """
        result = []
        for t, item in enumerate(cursor):
            if t < sample_k:
                result.append(item)
            else:
                m = self.rand.randint(0, t)
                if m < sample_k:
                    result[m] = item
        return result


def generate_train_data(data_set: DataSetEnum):
    logger.info("loading processed data...")
    all_data = load_processed_data(data_set)
    train, dev, test = split_train_dev_test_data(all_data)

    data_util = DataUtil(train, train, data_set, DataTypeEnum.Train)
    data_util.create_data_file()

    data_util = DataUtil(dev, pandas.concat([train, dev]), data_set, DataTypeEnum.Dev)
    data_util.create_data_file()

    data_util = DataUtil(test, all_data, data_set, DataTypeEnum.Test)
    data_util.create_data_file()


def main():
    overwrite = False
    for data_set in get_all_dataset():
        logger.info(f"-------- Creating train data for {data_set}... --------")
        train_data_path = TRAIN_DATA_PATH % (data_set, DataTypeEnum.Train)
        processed = ROOT_DIR.joinpath(train_data_path).exists()
        if not overwrite and processed:
            logger.warning(f"{data_set} is skipped because processed file already exist.")
            continue

        generate_train_data(data_set)


if __name__ == '__main__':
    main()
