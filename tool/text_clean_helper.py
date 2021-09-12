import re
from typing import Set

from tool.log_helper import logger
from tool.path_helper import ROOT_DIR
from tool.word2vec_helper import PAD_WORD


def get_stop_words(path="data/stopwords.txt") -> Set[str]:
    with open(ROOT_DIR.joinpath(path)) as f:
        return set(f.read().splitlines())


def get_punctuations(path="data/punctuations.txt") -> Set[str]:
    with open(ROOT_DIR.joinpath(path)) as f:
        return set(f.read().splitlines())


def tokenize(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def clean_text(review: str, max_len: int) -> (str, str):
    review = tokenize(review)
    review = review[:max_len]
    review += (max_len - len(review)) * [PAD_WORD]
    review = " ".join(review)
    return review


def test():
    text = "This is the third review of an irish album I write today (the others were Cranberries) " \
           "and now I'm sure about that Ireland is one of the countries producing the best music in the world. " \
           "And not just commercial pop-music in the Spice Girls way. Okay, " \
           "I just wanted to say something about Irish  music. Now let's say something about this album. " \
           "It's great. it's  beautiful. Very good, easy listened music. " \
           "If you like Enya or you just  want some easy-listened relaxing music. This is the album for you to buy!"
    logger.info(tokenize(text))
    logger.info(clean_text(text, 20)[1])


if __name__ == '__main__':
    test()
