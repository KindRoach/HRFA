import torch

from data_process.dataset.hrfa_self_dataset import HrfaSelfDataset
from model.base_model import BaseModel
from model.fm_layer import FMLayer
from model.hrfa import HrfaConfig, Attention, CnnEncoder
from tool.log_helper import logger
from tool.word2vec_helper import load_word_dict


class ReviewAttentionEncoder(torch.nn.Module):
    def __init__(self, config: HrfaConfig, self_id_count: int):
        super().__init__()
        self.config = config

        self.att_word = Attention(config, self_id_count)
        self.att_review = Attention(config, self_id_count)
        self.top_linear = torch.nn.Linear(config.kernel_num, config.latent_factors)
        self.dropout = torch.nn.Dropout(self.config.dropout)

    def forward(self, review, encoded_review, self_id):
        """
        :param review:          (batch size, review count, review length)
        :param encoded_review:  (batch size * review count, review length, kernel num)
        :param self_id:         (batch size, 1)
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

        # ------------------ Review Attention ------------------
        # review_att = [batch size, kernel num]
        mask = torch.all(review == self.config.pad_id, dim=2)
        review_att = self.att_review(reviews, self_id, mask)

        # latent = [batch size, latent factors]
        latent = self.dropout(torch.tanh(self.top_linear(review_att)))
        return latent


class HrfaSelfModel(BaseModel):
    def __init__(self, config: HrfaConfig, word_embedding_weight):
        super().__init__(config)

        assert config.kernel_width % 2 == 1

        word_embedding = torch.nn.Embedding.from_pretrained(word_embedding_weight, freeze=False, padding_idx=config.pad_id)

        self.user_encoder = CnnEncoder(config, word_embedding)
        self.user_att = ReviewAttentionEncoder(config, config.user_count)

        self.item_encoder = CnnEncoder(config, word_embedding)
        self.item_att = ReviewAttentionEncoder(config, config.item_count)

        self.fm_layer = FMLayer(2 * config.latent_factors, config.fm_k)
        self.loss_f = torch.nn.MSELoss()
        self.dataset_class = HrfaSelfDataset

        self.mid_var_map["user_gate"] = list()
        self.mid_var_map["item_gate"] = list()

    def forward(self, user_id, user_review, item_id, item_review):
        """
        :param user_id:                 (batch size, 1)
        :param user_review:             (batch size, review count, review length)

        :param item_id:                 (batch size, 1)
        :param item_review:             (batch size, review count, review length)

        :return:                        (batch size, 1)
        """

        encoded_user_review = self.user_encoder(user_review)
        user_latent = self.user_att(user_review, encoded_user_review, user_id)

        encoded_item_review = self.item_encoder(item_review)
        item_latent = self.item_att(item_review, encoded_item_review, item_id)

        predict = self.fm_layer(torch.cat([user_latent, item_latent], dim=1))
        return predict


def test():
    config = HrfaConfig()
    word_dict = load_word_dict(config.data_set)
    model = HrfaSelfModel(config, word_dict.weight)

    user_id = torch.randint(config.user_count, [config.batch_size, 1]).type(torch.long)
    user_review = torch.randint(config.word_count, [config.batch_size, config.review_count, config.review_length]).type(torch.long)

    item_id = torch.randint(config.item_count, [config.batch_size, 1]).type(torch.long)
    item_review = torch.randint(config.word_count, [config.batch_size, config.review_count, config.review_length]).type(torch.long)

    rating = torch.randint(5, [config.batch_size, 1])

    iter_i = (user_id, user_review,
              item_id, item_review,
              rating)

    predict, rating, loss = model.predict_iter_i(iter_i)
    logger.info(predict.shape)
    logger.info(rating.shape)
    logger.info(loss.shape)


if __name__ == '__main__':
    test()
