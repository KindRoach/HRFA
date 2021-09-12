from typing import List

import numpy as np
import torch
from gensim.models import Word2Vec
from pandas import DataFrame

from data_process.data_reader.data_enum import DataSetEnum
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR, create_path

PAD_WORD = "<PAD>"
UNK_WORD = "<UNK>"

WORD_EMBEDDING_SIZE = 50
WORD_DICT_PATH = "out/processed_data/%s/word_dict.pt"
UNK_WORDS_PATH = "out/processed_data/%s/UNKs.txt"


class WordDict:
    def __init__(self, emb_model: Word2Vec):
        self.emb = emb_model
        for word in [PAD_WORD, UNK_WORD]:
            if word not in emb_model:
                emb_model.add([word], np.zeros([1, emb_model.vector_size]))
                logger.info(f"Add {word} to word embedding.")

        self.weight = []
        self.word2idx = dict()
        self.idx2word = dict()
        self.add_words([PAD_WORD, UNK_WORD])

        # Make Sure idx(PAD_WORD) == 0
        assert self.word2idx[PAD_WORD] == 0

    def add_words(self, words: List[str]) -> List[str]:
        """
        :param words: List of words
        :return: Unknown words in pre-trained model.
        """
        unks = set()
        for word in words:
            if word not in self.word2idx:
                if word in self.emb:
                    idx = len(self.word2idx.keys())
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    self.weight.append(self.emb[word])
                else:
                    unks.add(word)
        return list(unks)

    def sentence2idx(self, s: str) -> List[int]:
        idx = []
        for word in s.split():
            if word in self.word2idx:
                idx.append(self.word2idx[word])
            else:
                idx.append(self.unk_id)
        return idx

    def save_dict(self, path: str):
        self.emb = None
        self.weight = torch.Tensor(self.weight)
        path = ROOT_DIR.joinpath(path)
        create_path(path)
        torch.save(self, path)
        logger.info("Word dict saved.")

    @property
    def unk_id(self):
        return self.word2idx[UNK_WORD]

    @property
    def pad_id(self):
        return self.word2idx[PAD_WORD]


def load_word_dict(data_set: DataSetEnum) -> WordDict:
    path = ROOT_DIR.joinpath(WORD_DICT_PATH % data_set)
    return torch.load(path)


def create_dict(name: str, df: DataFrame, emb_model: Word2Vec) -> WordDict:
    """
    Load all words from review text.
    """

    word_dict = WordDict(emb_model)
    unks = set()
    for review in df["review"]:
        unk = word_dict.add_words(review.split())
        unks.update(unk)

    # write unknown words to file.
    logger.warning(f"{len(unks)} unknown words!")
    path = ROOT_DIR.joinpath(UNK_WORDS_PATH % name)
    create_path(path)
    with open(path, "w", encoding="utf-8") as f:
        for word in unks:
            f.write(f"{word}\n")

    word_dict.save_dict(WORD_DICT_PATH % name)
    return word_dict


def test():
    sentences = [["hello", "good", "not_a_word"]]
    model = Word2Vec(sentences, min_count=1, size=64, workers=-1, window=1, sg=1, negative=64, iter=20).wv
    word_dict = WordDict(model)
    word_dict.add_words(["hello", "good", "not_a_word"])
    logger.info(word_dict.sentence2idx("hello , I'm good !"))
    word_dict.save_dict(WORD_DICT_PATH % "test")
    word_dict = load_word_dict(DataSetEnum.Test_Set)


if __name__ == "__main__":
    test()
