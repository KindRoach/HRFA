import math
from dataclasses import dataclass

import torch

from data_process.dataset.hrfa_dataset import HrfaDataset
from model.base_model import BaseModel, BaseConfig
from model.fm_layer import FMLayer
from tool.log_helper import logger
from tool.word2vec_helper import load_word_dict


@dataclass
class HrfaConfig(BaseConfig):
    max_rating = 5
    min_rating = 1
    kernel_width: int = 3
    kernel_num: int = 100
    latent_factors: int = 32
    id_dim: int = 32
    fm_k: int = 8


class CnnEncoder(torch.nn.Module):
    def __init__(self, config: HrfaConfig, word_emb):
        super().__init__()
        self.config = config
        self.word_emb = word_emb

        self.word_cnn = torch.nn.Conv1d(
            in_channels=config.word_dim,
            out_channels=config.kernel_num,
            kernel_size=config.kernel_width,
            padding=config.kernel_width // 2)

        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, review):
        """
        :param review:  (batch size, review count, review len)
        :return:        (batch size * review count, review length, kernel num)
        """

        # review_emb = [batch size, review count, review_length, word dim]
        review_emb = self.word_emb(review)

        # review_in_one = [batch size * review count, word dim, review len]
        review_in_one = review_emb.view(-1, self.config.review_length, self.config.word_dim)
        review_in_one = review_in_one.permute(0, 2, 1)

        # cnn_out = [batch size * review count, review length, kernel num]
        outputs = torch.relu(self.word_cnn(review_in_one))
        outputs = self.dropout(outputs).permute(0, 2, 1)

        return outputs


class Attention(torch.nn.Module):
    def __init__(self, config: HrfaConfig, id_count: int):
        super().__init__()

        self.id_embedding = torch.nn.Embedding(id_count, config.id_dim, padding_idx=config.pad_id)
        self.w_review = torch.nn.Linear(config.kernel_num, config.id_dim)
        self.w_id = torch.nn.Linear(config.id_dim, config.id_dim, bias=False)
        self.h = torch.nn.Linear(config.id_dim, 1)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, review, idx, mask):
        """
        :param review:  (batch size, review length, kernel num)
        :param idx:     (batch size, 1)
        :param mask:    (batch size, review length)
        :return:        (batch size, kernel num)
        """

        # idx_emb = [batch size, review length, latent factors]
        review_len = review.shape[1]
        idx_emb = self.id_embedding(idx).repeat(1, review_len, 1)

        # score = [batch size, review length]
        score = torch.relu(self.w_review(review) + self.w_id(idx_emb))
        score = self.dropout(self.h(score)).squeeze(2)
        score = score.masked_fill(mask, -1e10)
        score = torch.softmax(score, dim=1)

        # score = [batch size, 1, value len]
        score = score.unsqueeze(1)

        # out = [batch size, kernel num]
        out = torch.bmm(score, review).squeeze(1)

        return out


class ReviewAttentionEncoder(torch.nn.Module):
    def __init__(self, config: HrfaConfig, self_id_count: int):
        super().__init__()
        self.config = config

        self.att_word = Attention(config, self_id_count)
        self.att_review = Attention(config, self_id_count)
        self.top_linear = torch.nn.Linear(config.kernel_num, config.latent_factors)
        self.dropout = torch.nn.Dropout(self.config.dropout)

    def forward(self, review, encoded_review, self_id, self_rating, other_rating):
        """
        :param review:          (batch size, review count, review length)
        :param encoded_review:  (batch size * review count, review length, kernel num)
        :param self_id:         (batch size, 1)
        :param self_rating:     (batch size, review count)
        :param other_rating:    (batch size, review count)
        :return:                (batch size, latent factors)
        """

        # ------------------ Word Attention ------------------
        # mask = [batch size * review count, review length]
        mask = review == self.config.pad_id
        mask = mask.view(-1, self.config.review_length)

        # word_att = [batch size * review count, kernel num]
        idx_word = self_id.repeat(1, self.config.review_count)
        idx_word = idx_word.view(-1, 1)
        word_att = self.att_word(encoded_review, idx_word, mask)

        # reviews = [batch size, review count, kernel num]
        reviews = word_att.view(-1, self.config.review_count, self.config.kernel_num)

        # adjust for ref reviews
        if self_rating is not None and other_rating is not None:
            diff = (self_rating - other_rating) / (self.config.max_rating - self.config.min_rating)
            rating_sim = torch.cos(math.pi * diff)
            reviews = rating_sim.unsqueeze(2) * reviews

        # ------------------ Review Attention ------------------
        # review_att = [batch size, kernel num]
        mask = torch.all(review == self.config.pad_id, dim=2)
        review_att = self.att_review(reviews, self_id, mask)

        # latent = [batch size, latent factors]
        latent = self.dropout(torch.tanh(self.top_linear(review_att)))
        return latent


class CombineLayer(torch.nn.Module):
    def __init__(self, config: HrfaConfig):
        super().__init__()
        self.config = config
        self.w_self = torch.nn.Linear(config.latent_factors, 1)
        self.gate_weight = torch.nn.Parameter(torch.Tensor([math.exp(-i) for i in range(6)]))
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, self_review_count, self_latent, ref_latent):
        """
        :param self_review_count:   (batch size, 1)
        :param self_latent:         (batch size, latent factors)
        :param ref_latent:          (batch size, latent factors)
        :return:                    (batch size, latent factors), (batch size, 1)
        """

        w_s = self.dropout(self.w_self(self_latent))
        greater_than_5 = self_review_count > 5
        self_review_count = self_review_count.masked_fill(greater_than_5, 5)
        gate = torch.sigmoid(self.gate_weight[self_review_count] * w_s)

        # comb_latent = [batch size, latent factors]
        comb_latent = gate * ref_latent + (1 - gate) * self_latent
        return comb_latent, gate


class HrfaModel(BaseModel):
    def __init__(self, config: HrfaConfig, word_embedding_weight):
        super().__init__(config)

        assert config.kernel_width % 2 == 1

        word_embedding = torch.nn.Embedding.from_pretrained(word_embedding_weight, freeze=False, padding_idx=config.pad_id)

        self.user_encoder = CnnEncoder(config, word_embedding)
        self.user_att = ReviewAttentionEncoder(config, config.user_count)
        self.user_ref_att = ReviewAttentionEncoder(config, config.user_count)

        self.item_encoder = CnnEncoder(config, word_embedding)
        self.item_att = ReviewAttentionEncoder(config, config.item_count)
        self.item_ref_att = ReviewAttentionEncoder(config, config.item_count)

        self.user_comb_layer = CombineLayer(config)
        self.item_comb_layer = CombineLayer(config)
        self.fm_layer = FMLayer(2 * config.latent_factors, config.fm_k)
        self.loss_f = torch.nn.MSELoss()
        self.dataset_class = HrfaDataset

        self.mid_var_map["user_gate"] = list()
        self.mid_var_map["item_gate"] = list()

    def forward(self, user_id, user_review,
                item_id, item_review,
                user_ref_review, user_ref_self_rating, user_ref_other_rating,
                item_ref_review, item_ref_self_rating, item_ref_other_rating):
        """
        :param user_id:                 (batch size, 1)
        :param user_review:             (batch size, review count, review length)

        :param item_id:                 (batch size, 1)
        :param item_review:             (batch size, review count, review length)

        :param user_ref_review:         (batch size, review count, review length)
        :param user_ref_self_rating:    (batch size, review count)
        :param user_ref_other_rating:   (batch size, review count)

        :param item_ref_review:         (batch size, review count, review length)
        :param item_ref_self_rating:    (batch size, review count)
        :param item_ref_other_rating:   (batch size, review count)
        :return:                        (batch size, 1)
        """

        encoded_user_review = self.user_encoder(user_review)
        user_latent = self.user_att(user_review, encoded_user_review, user_id, None, None)

        encoded_item_review = self.item_encoder(item_review)
        item_latent = self.item_att(item_review, encoded_item_review, item_id, None, None)

        encoded_user_ref_review = self.user_encoder(user_ref_review)
        user_ref_latent = self.user_ref_att(user_ref_review, encoded_user_ref_review, user_id, user_ref_self_rating, user_ref_other_rating)

        encoded_item_ref_review = self.item_encoder(item_ref_review)
        item_ref_latent = self.item_ref_att(item_ref_review, encoded_item_ref_review, item_id, item_ref_self_rating, item_ref_other_rating)

        user_self_review_count = torch.any(user_review != self.config.pad_id, dim=2).sum(dim=1).unsqueeze(1)
        user_comb_latent, user_gate = self.user_comb_layer(user_self_review_count, user_latent, user_ref_latent)

        if self.record_mid_var:
            self.mid_var_map["user_gate"].append(user_gate)

        item_self_review_count = torch.any(item_review != self.config.pad_id, dim=2).sum(dim=1).unsqueeze(1)
        item_comb_latent, item_gate = self.item_comb_layer(item_self_review_count, item_latent, item_ref_latent)

        if self.record_mid_var:
            self.mid_var_map["item_gate"].append(item_gate)

        predict = self.fm_layer(torch.cat([user_comb_latent, item_comb_latent], dim=1))
        return predict


def test():
    config = HrfaConfig()
    word_dict = load_word_dict(config.data_set)
    model = HrfaModel(config, word_dict.weight)

    user_id = torch.randint(config.user_count, [config.batch_size, 1]).type(torch.long)
    user_review = torch.randint(config.word_count, [config.batch_size, config.review_count, config.review_length]).type(torch.long)

    item_id = torch.randint(config.item_count, [config.batch_size, 1]).type(torch.long)
    item_review = torch.randint(config.word_count, [config.batch_size, config.review_count, config.review_length]).type(torch.long)

    user_ref_review = torch.randint(config.word_count, [config.batch_size, config.review_count, config.review_length]).type(torch.long)
    user_ref_self_rating = torch.randint(5, [config.batch_size, config.review_count]).type(torch.float32) + 1
    user_ref_other_rating = torch.randint(5, [config.batch_size, config.review_count]).type(torch.float32) + 1

    item_ref_review = torch.randint(config.word_count, [config.batch_size, config.review_count, config.review_length]).type(torch.long)
    item_ref_self_rating = torch.randint(5, [config.batch_size, config.review_count]).type(torch.float32) + 1
    item_ref_other_rating = torch.randint(5, [config.batch_size, config.review_count]).type(torch.float32) + 1

    rating = torch.randint(5, [config.batch_size, 1])

    iter_i = (user_id, user_review,
              item_id, item_review,
              user_ref_review, user_ref_self_rating, user_ref_other_rating,
              item_ref_review, item_ref_self_rating, item_ref_other_rating,
              rating)

    predict, rating, loss = model.predict_iter_i(iter_i)
    logger.info(predict.shape)
    logger.info(rating.shape)
    logger.info(loss.shape)


if __name__ == '__main__':
    test()
