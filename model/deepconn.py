from dataclasses import dataclass

import torch

from data_process.dataset.deepconn_dataset import DeepConnDataset
from model.base_model import BaseConfig, BaseModel
from model.fm_layer import FMLayer
from tool.log_helper import logger
from tool.word2vec_helper import load_word_dict


@dataclass
class DeepConnConfig(BaseConfig):
    kernel_width: int = 3
    kernel_num: int = 100
    latent_factors: int = 32
    fm_k: int = 8


class ConvMaxLayer(torch.nn.Module):
    """
    The independent layer for user review and item review.
    """

    def __init__(self, config: DeepConnConfig):
        super().__init__()
        self.config = config

        self.conv = torch.nn.Conv1d(
            in_channels=config.word_dim,
            out_channels=config.kernel_num,
            kernel_size=config.kernel_width)

        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=config.review_length - config.kernel_width + 1,
            stride=1)

        self.full_connect = torch.nn.Linear(config.kernel_num, config.latent_factors)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, review):
        """
        :param review:  (batch size, review length, word dim)
        :return:        (batch size, latent factors)
        """

        review = review.permute(0, 2, 1)

        # out = [batch size, kernel num, review length - kernel width + 1]
        out = torch.relu(self.conv(review))
        out = self.dropout(out)

        # max_out = [batch size, kernel num]
        max_out = self.max_pool(out)
        max_out = max_out.squeeze(2)

        # latent = [batch size, latent factors]
        latent = self.full_connect(max_out)
        latent = self.dropout(latent)

        return latent


class DeepConnModel(BaseModel):
    """
    Main network, including two independent ConvMaxLayers and one shared FMLayer.
    """

    def __init__(self, config: DeepConnConfig, word_embedding_weight):
        super().__init__(config)
        assert config.review_count == 1

        self.embedding = torch.nn.Embedding.from_pretrained(word_embedding_weight, freeze=False, padding_idx=config.pad_id)
        self.user_layer = ConvMaxLayer(config)
        self.item_layer = ConvMaxLayer(config)
        self.fm_layer = FMLayer(config.latent_factors * 2, config.fm_k)

        self.loss_f = torch.nn.MSELoss()
        self.dataset_class = DeepConnDataset

    def forward(self, user_review, item_review):
        """
        :param user_review: (batch size, review length)
        :param item_review: (batch size, review length)
        :return:            (batch size, 1)
        """

        # *_review = [batch size, review length, word dim]
        user_review = self.embedding(user_review)
        item_review = self.embedding(item_review)

        # *_latent = [batch size, latent factors]
        user_latent = self.user_layer(user_review)
        item_latent = self.item_layer(item_review)

        # predict = [batch size, 1]
        latent = torch.cat([user_latent, item_latent], dim=1)
        predict = self.fm_layer(latent)
        return predict


def test():
    config = DeepConnConfig(review_count=1)
    word_dict = load_word_dict(config.data_set)
    model = DeepConnModel(config, word_dict.weight)

    review_size = [config.batch_size, config.review_length]
    user_review = torch.randint(config.word_count, review_size).type(torch.long)
    item_review = torch.randint(config.word_count, review_size).type(torch.long)
    rating = torch.randint(5, [config.batch_size, 1])

    iter_i = (user_review, item_review, rating)
    predict, rating, loss = model.predict_iter_i(iter_i)
    logger.info(predict.shape)
    logger.info(rating.shape)
    logger.info(loss.shape)


if __name__ == '__main__':
    test()
