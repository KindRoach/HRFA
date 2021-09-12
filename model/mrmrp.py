from dataclasses import dataclass

import torch

from data_process.dataset.mrmrp_dataset import MrmrpDataset
from model.base_model import BaseModel
from model.deepconn import ConvMaxLayer, DeepConnConfig
from model.mlp import MlpLayer
from tool.log_helper import logger
from tool.word2vec_helper import load_word_dict


@dataclass
class MrmrpConfig(DeepConnConfig):
    mlp_layers: int = 3


class CombineLayer(torch.nn.Module):
    def __init__(self, config: MrmrpConfig):
        super().__init__()
        self.config = config

        self.w_u = torch.nn.Linear(config.latent_factors, config.latent_factors)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, user_latent, item_latent, sup_latent):
        """
        :param user_latent: (batch size, latent factors)
        :param item_latent: (batch size, latent factors)
        :param sup_latent: (batch size, latent factors)
        :return: (batch size, 3 * latent factors)
        """

        # latent_us = [batch size, latent factors]
        gate = self.dropout(self.w_u(user_latent))
        latent_us = gate * sup_latent

        # latent_comb = [batch size, 3 * latent factors]
        latent_comb = torch.cat((user_latent, item_latent, latent_us), dim=1)
        return latent_comb


class PredictLayer(torch.nn.Module):
    def __init__(self, config: MrmrpConfig):
        super().__init__()
        self.config = config

        self.mlp_layer = MlpLayer(config.mlp_layers, config.latent_factors * 3, 1, config.dropout, "relu")
        self.b_user = torch.nn.Parameter(torch.zeros([config.user_count]), requires_grad=True)
        self.b_item = torch.nn.Parameter(torch.zeros([config.item_count]), requires_grad=True)

    def forward(self, feature, user_id, item_id):
        """
        :param feature: (batch size, 1)
        :param user_id: (batch size, 1)
        :param item_id: (batch size, 1)
        :return:        (batch size, 1)
        """

        feature = self.mlp_layer(feature)
        predict = feature + self.b_user[user_id] + self.b_item[item_id] + self.config.avg_rating
        return predict


class MrmrpModel(BaseModel):

    def __init__(self, config: MrmrpConfig, word_embedding_weight):
        super().__init__(config)
        assert config.review_count == 1
        assert config.kernel_width % 2 == 1

        self.embedding = torch.nn.Embedding.from_pretrained(word_embedding_weight, freeze=False, padding_idx=config.pad_id)
        self.user_layer = ConvMaxLayer(config)
        self.item_layer = ConvMaxLayer(config)
        self.sup_layer = ConvMaxLayer(config)
        self.comb_layer = CombineLayer(config)
        self.predict_layer = PredictLayer(config)

        self.loss_f = torch.nn.MSELoss()
        self.dataset_class = MrmrpDataset

    def forward(self, user_review, item_review, sup_review, user_id, item_id):
        """
        :param user_review: (batch size, review length)
        :param item_review: (batch size, review length)
        :param sup_review:  (batch size, review length)
        :param user_id:     (batch size, 1)
        :param item_id:     (batch size, 1)
        :return:            (batch size, 1)
        """

        # *_review = [batch size, review length, word dim]
        user_review = self.embedding(user_review)
        item_review = self.embedding(item_review)
        sup_review = self.embedding(sup_review)

        # *_latent = [batch size, latent factors]
        user_latent = self.user_layer(user_review)
        item_latent = self.item_layer(item_review)
        sup_latent = self.sup_layer(sup_review)

        comb_latent = self.comb_layer(user_latent, item_latent, sup_latent)

        # predict = [batch size, 1]
        predict = self.predict_layer(comb_latent, user_id, item_id)
        return predict


def test():
    config = MrmrpConfig(review_count=1)
    word_dict = load_word_dict(config.data_set)
    model = MrmrpModel(config, word_dict.weight)

    review_size = [config.batch_size, config.review_length]
    user_review = torch.randint(config.word_count, review_size).type(torch.long)
    item_review = torch.randint(config.word_count, review_size).type(torch.long)
    sup_review = torch.randint(config.word_count, review_size).type(torch.long)
    user_id = torch.randint(config.user_count, [config.batch_size, 1]).type(torch.long)
    item_id = torch.randint(config.user_count, [config.batch_size, 1]).type(torch.long)
    rating = torch.randint(5, [config.batch_size, 1])

    iter_i = (user_review, item_review, sup_review, rating, user_id, item_id)
    predict, rating, loss = model.predict_iter_i(iter_i)
    logger.info(predict.shape)
    logger.info(rating.shape)
    logger.info(loss.shape)


if __name__ == '__main__':
    test()
