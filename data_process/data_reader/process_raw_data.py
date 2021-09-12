import json
import multiprocessing
import pickle
import random

import pandas
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_process.data_reader.data_enum import DataSetEnum, RANDOM_STATE, REVIEW_LENGTH, get_all_dataset
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR, create_path
from tool.text_clean_helper import clean_text
from tool.word2vec_helper import create_dict, WORD_EMBEDDING_SIZE, PAD_WORD, UNK_WORD

MAX_DATA_SIZE = 1000000

PROCESSED_PATH = "out/processed_data/%s/reviews.pt"
ORIGINAL_PATH = "data/raw_data/%s.json"
SENTENCE_FILE_PATH = "out/processed_data/%s/sentences.txt"


def split_train_dev_test_data(data: DataFrame) -> (DataFrame, DataFrame, DataFrame):
    train, test = train_test_split(data, test_size=1 / 10, random_state=RANDOM_STATE)
    train, dev = train_test_split(train, test_size=1 / 9, random_state=RANDOM_STATE)
    return train, dev, test


def get_file_line_number(in_path: str):
    line_number = 0
    with open(ROOT_DIR.joinpath(in_path), "r", encoding="utf-8") as in_f:
        for line in in_f:
            line_number += 1
    return line_number


def process_raw_data(data_set: DataSetEnum):
    """
    Read raw data and remove useless columns and clear review text.
    Then save the result to file system.
    """

    logger.info(f"reading raw data: {data_set}...")

    reviews = []
    ui_set = set()
    line_number = get_file_line_number(ORIGINAL_PATH % data_set)
    with open(ROOT_DIR.joinpath(ORIGINAL_PATH % data_set), "r", encoding="utf-8") as in_f:
        if line_number > MAX_DATA_SIZE:
            logger.warning(f"Data is large than {MAX_DATA_SIZE} and will be randomly sampled.")
        r = random.Random(RANDOM_STATE)
        for line in tqdm(in_f, desc="Read finished", total=line_number):
            json_obj = json.loads(line)

            # skip None or empty.
            if any([
                "reviewerID" not in json_obj,
                "asin" not in json_obj,
                "reviewText" not in json_obj,
                "overall" not in json_obj
            ]):
                continue

            review = {
                "rawUserID": json_obj["reviewerID"],
                "rawItemID": json_obj["asin"],
                "review": json_obj["reviewText"],
                "rating": json_obj["overall"],
            }

            # skip None or empty.
            if any([
                not review["rawUserID"],
                not review["rawItemID"],
                not review["review"]
            ]):
                continue

            # distinct on userID and itemID
            ui_tuple = (review["rawUserID"], review["rawItemID"])
            if ui_tuple not in ui_set:
                ui_set.add(ui_tuple)
            else:
                continue

            # Sample
            if len(reviews) < MAX_DATA_SIZE:
                # Text clean
                review["review"] = clean_text(review["review"], REVIEW_LENGTH)
                reviews.append(review)
            else:
                i = r.randint(0, len(reviews))
                if i < MAX_DATA_SIZE:
                    # Text clean
                    review["review"] = clean_text(review["review"], REVIEW_LENGTH)
                    reviews[i] = review

    df = pandas.DataFrame(reviews)

    # assign new numeric id to user and item.
    # new id start from 1, leaving 0 for pad.
    df["userID"] = df.groupby(df["rawUserID"]).ngroup() + 1
    df["itemID"] = df.groupby(df["rawItemID"]).ngroup() + 1

    return df


def load_processed_data(data_set: DataSetEnum):
    """
    Columns includes ["userID", "itemID", "review", "rating"] at least.
    """
    path = ROOT_DIR.joinpath(PROCESSED_PATH % data_set)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_processed_data(data: DataFrame, data_set: DataSetEnum):
    path = ROOT_DIR.joinpath(PROCESSED_PATH % data_set)
    create_path(path)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Processed data saved.")


def convert_text_to_idx(all_data, data_set, g_model):
    logger.info("Converting review text to idx...")
    word_dict = create_dict(data_set.value, all_data, g_model)
    all_data["review"] = all_data["review"].apply(word_dict.sentence2idx)
    logger.info("Text converted.")


def get_user_count(data: DataFrame) -> int:
    return max(data["userID"]) + 1


def get_item_count(data: DataFrame) -> int:
    return max(data["itemID"]) + 1


def get_max_rating(data: DataFrame) -> int:
    return max(data["rating"])


def get_avg_rating(data: DataFrame) -> float:
    return data["rating"].mean()


def get_max_review_length(data: DataFrame) -> int:
    review_lengths = data["review"].apply(lambda review: len(review))
    max_length = review_lengths.max()
    return max_length


def get_max_review_count(data: DataFrame):
    review_count_user = data["review"].groupby([data["userID"]]).count()
    review_count_user = review_count_user.max()

    review_count_item = data["review"].groupby([data["itemID"]]).count()
    review_count_item = review_count_item.max()

    max_count = max(review_count_item, review_count_user)
    return max_count


def train_gensim_model(data_set: DataSetEnum):
    logger.info("training gensim model...")
    sentences = LineSentence(ROOT_DIR.joinpath(SENTENCE_FILE_PATH % data_set))
    g_model = Word2Vec(sentences, size=WORD_EMBEDDING_SIZE, workers=multiprocessing.cpu_count(), sg=1).wv
    logger.info("gensim model trained.")
    return g_model


def save_sentences(all_data: DataFrame, data_set: DataSetEnum):
    logger.info("saving sentences...")
    sentences_path = ROOT_DIR.joinpath(SENTENCE_FILE_PATH % data_set)
    create_path(sentences_path)

    with open(sentences_path, "w", encoding="utf-8") as out_f:
        for review in all_data["review"]:
            words = [w for w in review.split() if w not in [PAD_WORD, UNK_WORD]]
            out_f.write(f"{' '.join(words)}\n")


def main():
    overwrite = False
    for data_set in get_all_dataset():
        logger.info(f"------------Processing dataset {data_set}------------")
        if not overwrite and ROOT_DIR.joinpath(PROCESSED_PATH % data_set).exists():
            logger.warning(f"{data_set} is skipped because processed file already exist.")
            continue

        all_data = process_raw_data(data_set)
        save_sentences(all_data, data_set)
        g_model = train_gensim_model(data_set)
        convert_text_to_idx(all_data, data_set, g_model)
        save_processed_data(all_data, data_set)

        logger.info(f"Max user id = {get_user_count(all_data)}")
        logger.info(f"Max item id = {get_item_count(all_data)}")

        logger.info(f"Max review length = {get_max_review_length(all_data)}")
        logger.info(f"Max review count = {get_max_review_count(all_data)}")


if __name__ == '__main__':
    main()
