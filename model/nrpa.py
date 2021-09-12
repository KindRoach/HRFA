from dataclasses import dataclass

import torch

from data_process.dataset.nrpa_dataset import NrpaDataset
from model.base_model import BaseModel, BaseConfig
from model.fm_layer import FMLayer
from tool.log_helper import logger
from tool.word2vec_helper import load_word_dict


@dataclass
class NrpaConfig(BaseConfig):
    kernel_width: int = 3
    kernel_num: int = 100
    id_dim: int = 32
    fm_k: int = 8


class Attention(torch.nn.Module):
    def __init__(self, config: NrpaConfig, id_count: int):
        super().__init__()
        self.config = config

        self.id_embedding = torch.nn.Embedding(id_count, config.id_dim, padding_idx=config.pad_id)
        self.query_linear = torch.nn.Linear(config.id_dim, config.kernel_num)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, review, idx, mask):
        """
        :param review:  (batch size, kernel num, review length)
        :param idx:     (batch size, 1)
        :param mask:    (batch size, review length)
        :return:        (batch size, kernel num)
        """

        # idx_emb = [batch size, latent factors]
        idx_emb = self.id_embedding(idx).squeeze(1)

        # query = [batch size, kernel num]
        query = torch.relu(self.query_linear(idx_emb))
        query = self.dropout(query)

        # query = [batch size, 1, kernel num]
        query = query.unsqueeze(1)

        # score = [batch size, review length]
        score = torch.bmm(query, review).squeeze(1)
        score = score.masked_fill(mask, -1e10)
        score = torch.softmax(score, dim=1)

        # score = [batch size, 1, value len]
        score = score.unsqueeze(1)

        # out = [batch size, kernel num]
        review = review.permute(0, 2, 1)
        out = torch.bmm(score, review).squeeze(1)

        return out


class ReviewEncoder(torch.nn.Module):
    def __init__(self, config: NrpaConfig, word_emb, id_count: int):
        super().__init__()
        self.config = config
        self.word_emb = word_emb

        self.conv = torch.nn.Conv1d(
            in_channels=config.word_dim,
            out_channels=config.kernel_num,
            kernel_size=config.kernel_width,
            padding=config.kernel_width // 2)

        self.att_word = Attention(config, id_count)
        self.att_review = Attention(config, id_count)
        self.dropout = torch.nn.Dropout(self.config.dropout)

    def forward(self, review, idx):
        """
        :param review:  (batch size, review count, review length)
        :param idx:     (batch size, 1)
        :return:        (batch size, kernel num)
        """

        # review_in_one = [batch size * review count, word dim, review_length]
        review_emb = self.word_emb(review)
        batch_size = review_emb.shape[0]
        review_in_one = review_emb.view(-1, self.config.review_length, self.config.word_dim)
        review_in_one = review_in_one.permute(0, 2, 1)

        # word_cnn = [batch size * review count, review length, kernel num]
        word_cnn = torch.relu(self.conv(review_in_one))
        word_cnn = self.dropout(word_cnn)

        # ------------------ Word Attention ------------------
        # mask = [batch size * review count, review length]
        mask = review == self.config.pad_id
        mask = mask.view(-1, self.config.review_length)

        # mask = [batch size * review count, 1]
        idx_word = idx.repeat(1, self.config.review_count)
        idx_word = idx_word.view(-1, 1)

        # word_att = [batch size * review count, kernel num]
        word_att = self.att_word(word_cnn, idx_word, mask)

        # reviews = [batch size, kernel num, review count]
        reviews = word_att.view(-1, self.config.review_count, self.config.kernel_num)
        reviews = reviews.permute(0, 2, 1)

        # ------------------ Review Attention ------------------
        # review_att = [batch size, kernel num]
        mask = torch.all(review == self.config.pad_id, dim=2)
        review_att = self.att_review(reviews, idx, mask)
        return review_att


class NrpaModel(BaseModel):
    def __init__(self, config: NrpaConfig, word_embedding_weight):
        super().__init__(config)

        assert config.kernel_width % 2 == 1

        word_embedding = torch.nn.Embedding.from_pretrained(word_embedding_weight, freeze=False, padding_idx=config.pad_id)
        self.user_review_layer = ReviewEncoder(config, word_embedding, config.user_count)
        self.item_review_layer = ReviewEncoder(config, word_embedding, config.item_count)
        self.fm_layer = FMLayer(2 * config.kernel_num, config.fm_k)

        self.loss_f = torch.nn.MSELoss()
        self.dataset_class = NrpaDataset

    def forward(self, user_id, user_review, item_id, item_review):
        """
        :param user_id:             (batch size, 1)
        :param user_review:         (batch size, review count, review length)
        :param item_id:             (batch size, 1)
        :param item_review:         (batch size, review count, review length)
        :return:                    (batch size, 1)
        """

        user_feature = self.user_review_layer(user_review, user_id)
        item_feature = self.item_review_layer(item_review, item_id)
        predict = self.fm_layer(torch.cat([user_feature, item_feature], dim=1))
        return predict


def test():
    config = NrpaConfig()
    word_dict = load_word_dict(config.data_set)
    model = NrpaModel(config, word_dict.weight)

    user_id = torch.randint(config.user_count, [config.batch_size, 1]).type(torch.long)
    user_review = torch.randint(config.word_count, [config.batch_size, config.review_count, config.review_length]).type(torch.long)

    item_id = torch.randint(config.item_count, [config.batch_size, 1]).type(torch.long)
    item_review = torch.randint(config.word_count, [config.batch_size, config.review_count, config.review_length]).type(torch.long)

    rating = torch.randint(5, [config.batch_size, 1])

    iter_i = (user_id, user_review, item_id, item_review, rating)
    predict, rating, loss = model.predict_iter_i(iter_i)
    logger.info(predict.shape)
    logger.info(rating.shape)
    logger.info(loss.shape)


if __name__ == '__main__':
    test()
