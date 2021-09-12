from dataclasses import dataclass

import torch

from data_process.dataset.parl_dataset import ParlDataset
from model.base_model import BaseModel
from model.deepconn import ConvMaxLayer, DeepConnConfig
from model.fm_layer import FMLayerWithBias
from tool.log_helper import logger
from tool.word2vec_helper import load_word_dict


@dataclass
class ParlConfig(DeepConnConfig):
    aux_loss_weight: float = 0.01


class AuxiliaryLayer(torch.nn.Module):
    def __init__(self, config: ParlConfig):
        super().__init__()
        self.config = config

        self.init_conv = torch.nn.Conv1d(
            in_channels=config.word_dim,
            out_channels=config.kernel_num,
            kernel_size=config.kernel_width,
            padding=config.kernel_width // 2)

        self.abstract_conv = torch.nn.Conv1d(
            in_channels=config.kernel_num,
            out_channels=config.kernel_num,
            kernel_size=config.kernel_width)

        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=config.review_length - config.kernel_width + 1,
            stride=1)

        self.aux_linear = torch.nn.Linear(config.kernel_num, config.latent_factors)
        self.high_linear = torch.nn.Linear(config.latent_factors, config.latent_factors)
        self.high_gate_linear = torch.nn.Linear(config.latent_factors, 1)

        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, review):
        """
        :param review:  (batch size, review length, word dim)
        :return:        (batch size, latent factors)
        """

        review = review.permute(0, 2, 1)

        # out_c = [batch size, kernel num, review length]
        out_c = torch.relu(self.init_conv(review))
        out_c = self.dropout(out_c)

        # out_q = [batch size, kernel num, review length - kernel width + 1]
        out_q = torch.relu(self.abstract_conv(out_c))
        out_q = self.dropout(out_q)

        # q_abs = [batch size, kernel num]
        q_abs = self.max_pool(out_q)
        q_abs = q_abs.squeeze(2)

        # q_abs = [batch size, latent factors]
        q_aux = torch.relu(self.aux_linear(q_abs))
        q_aux = self.dropout(q_aux)

        # q_high = [batch size, latent factors]
        q_high = torch.relu(self.high_linear(q_aux))
        q_high = self.dropout(q_high)

        # q_high = [batch size, latent factors]
        miu = torch.sigmoid(self.high_gate_linear(q_high))
        miu = self.dropout(miu)
        q_high = miu * q_high + (1 - miu) * q_aux

        return q_high


class CombineLayer(torch.nn.Module):
    def __init__(self, config: ParlConfig):
        super().__init__()
        self.config = config

        self.w_u = torch.nn.Linear(config.latent_factors, config.latent_factors)
        self.w_i = torch.nn.Linear(config.latent_factors, config.latent_factors, bias=False)
        self.w_aux = torch.nn.Linear(config.latent_factors, config.latent_factors, bias=False)

        self.w_comb = torch.nn.Linear(config.latent_factors * 2, config.latent_factors)

        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, user_latent, item_latent, aux_latent):
        """
        :param user_latent: (batch size, latent factors)
        :param item_latent: (batch size, latent factors)
        :param aux_latent: (batch size, latent factors)
        :return: (batch size, 2 * latent factors)
        """

        # t_aux = [batch size, latent factors]
        u = self.w_u(user_latent)
        i = self.w_i(item_latent)
        aux = self.w_aux(aux_latent)
        gt = self.dropout(torch.tanh(u + i + aux))
        t_aux = gt * aux_latent

        # t_comb = [batch size, latent factors]
        t_comb = torch.cat((user_latent, t_aux), dim=1)
        t_comb = torch.relu(self.w_comb(t_comb))

        # z_ui = [batch size, 2 * latent factors]
        z_ui = torch.cat((t_comb, item_latent), dim=1)
        return z_ui


class ParlModel(BaseModel):

    def __init__(self, config: ParlConfig, word_embedding_weight):
        super().__init__(config)
        assert config.review_count == 1
        assert config.kernel_width % 2 == 1

        self.embedding = torch.nn.Embedding.from_pretrained(word_embedding_weight, freeze=False, padding_idx=config.pad_id)
        self.user_layer = ConvMaxLayer(config)
        self.item_layer = ConvMaxLayer(config)
        self.auxiliary_layer = AuxiliaryLayer(config)
        self.comb_layer = CombineLayer(config)
        self.prediction_layer = FMLayerWithBias(config.latent_factors * 2, config.fm_k, config.user_count, config.item_count)

        self.loss_f = torch.nn.MSELoss()
        self.dataset_class = ParlDataset

    def forward(self, user_review, item_review, aux_review, user_id, item_id):
        """
        :param user_review: (batch size, review length)
        :param item_review: (batch size, review length)
        :param aux_review:  (batch size, review length)
        :param user_id:     (batch size, 1)
        :param item_id:     (batch size, 1)
        :return:            (batch size, 1), (batch size, 1)
        """

        # *_review = [batch size, review length, word dim]
        user_review = self.embedding(user_review)
        item_review = self.embedding(item_review)
        aux_review = self.embedding(aux_review)

        # *_latent = [batch size, latent factors]
        user_latent = self.user_layer(user_review)
        item_latent = self.item_layer(item_review)
        aux_latent = self.auxiliary_layer(aux_review)

        aux_loss = (user_latent - aux_latent).square().sum(dim=1).mean()
        aux_loss = aux_loss * self.config.aux_loss_weight
        latent = self.comb_layer(user_latent, item_latent, aux_latent)

        # predict = [batch size, 1]
        predict = self.prediction_layer(latent, user_id, item_id)
        return predict, aux_loss

    def predict_iter_i(self, iter_i):
        iter_i = self.move_iter_i_to_device(iter_i)
        args = iter_i[:-1]
        rating = iter_i[-1]
        predict, aux_loss = self(*args)
        loss = self.loss_f(predict, rating) + aux_loss
        return predict, rating, loss


def test():
    config = ParlConfig(review_count=1)
    word_dict = load_word_dict(config.data_set)
    model = ParlModel(config, word_dict.weight)

    review_size = [config.batch_size, config.review_length]
    user_review = torch.randint(config.word_count, review_size).type(torch.long)
    item_review = torch.randint(config.word_count, review_size).type(torch.long)
    aux_review = torch.randint(config.word_count, review_size).type(torch.long)
    user_id = torch.randint(config.user_count, [config.batch_size, 1]).type(torch.long)
    item_id = torch.randint(config.user_count, [config.batch_size, 1]).type(torch.long)
    rating = torch.randint(5, [config.batch_size, 1])

    iter_i = (user_review, item_review, aux_review, user_id, item_id, rating)
    predict, rating, loss = model.predict_iter_i(iter_i)
    logger.info(predict.shape)
    logger.info(rating.shape)
    logger.info(loss.shape)


if __name__ == '__main__':
    test()
