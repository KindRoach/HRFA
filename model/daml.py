from dataclasses import dataclass

import torch

from data_process.dataset.anr_dataset import AnrDataset
from model.base_model import BaseConfig, BaseModel
from model.mlp import MlpLayer
from tool.log_helper import logger
from tool.word2vec_helper import load_word_dict


@dataclass
class DamlConfig(BaseConfig):
    kernel_width: int = 3
    kernel_num: int = 100
    latent_factors: int = 32
    mlp_layers: int = 1


class DamlModel(BaseModel):
    """
    KDD 2019 DAML
    """

    def __init__(self, config: DamlConfig, word_embedding_weight):
        super().__init__(config)

        self.opt = config
        self.num_fea = 2  # ID + DOC
        self.user_word_embs = torch.nn.Embedding.from_pretrained(word_embedding_weight, freeze=False, padding_idx=config.pad_id)
        self.item_word_embs = torch.nn.Embedding.from_pretrained(word_embedding_weight, freeze=False, padding_idx=config.pad_id)

        # share
        self.word_cnn = torch.nn.Conv2d(1, 1, (5, config.word_dim), padding=(2, 0))
        # document-level cnn
        self.user_doc_cnn = torch.nn.Conv2d(1, config.kernel_num, (config.kernel_width, config.word_dim), padding=(1, 0))
        self.item_doc_cnn = torch.nn.Conv2d(1, config.kernel_num, (config.kernel_width, config.word_dim), padding=(1, 0))
        # abstract-level cnn
        self.user_abs_cnn = torch.nn.Conv2d(1, config.kernel_num, (config.kernel_width, config.kernel_num))
        self.item_abs_cnn = torch.nn.Conv2d(1, config.kernel_num, (config.kernel_width, config.kernel_num))

        self.unfold = torch.nn.Unfold((3, config.kernel_num), padding=(1, 0))

        # fc layer
        self.user_fc = torch.nn.Linear(config.kernel_num, config.latent_factors)
        self.item_fc = torch.nn.Linear(config.kernel_num, config.latent_factors)

        self.uid_embedding = torch.nn.Embedding(config.user_count + 2, config.latent_factors)
        self.iid_embedding = torch.nn.Embedding(config.item_count + 2, config.latent_factors)
        self.predict_layer = PredictLayer(config)

        self.loss_f = torch.nn.MSELoss()
        self.dataset_class = AnrDataset

        self.reset_para()

    def forward(self, user_review, item_review, user_id, item_id):
        """
        :param user_review: (batch size, review length)
        :param item_review: (batch size, review length)
        :param user_id:     (batch size, 1)
        :param item_id:     (batch size, 1)
        :return:            (batch size, 1)
        """

        # ------------------ review encoder ---------------------------------
        user_word_embs = self.user_word_embs(user_review)
        item_word_embs = self.item_word_embs(item_review)
        # (BS, 100, DOC_LEN, 1)
        user_local_fea = self.local_attention_cnn(user_word_embs, self.user_doc_cnn)
        item_local_fea = self.local_attention_cnn(item_word_embs, self.item_doc_cnn)

        # DOC_LEN * DOC_LEN
        euclidean = torch.norm(user_local_fea - item_local_fea, p=2, dim=1)
        attention_matrix = 1.0 / (1 + euclidean)

        # (?, DOC_LEN)
        attention = attention_matrix.squeeze(2)

        # (?, 32)
        user_doc_fea = self.local_pooling_cnn(user_local_fea, attention, self.user_abs_cnn, self.user_fc)
        item_doc_fea = self.local_pooling_cnn(item_local_fea, attention, self.item_abs_cnn, self.item_fc)

        # ------------------ id embedding ---------------------------------
        uid_emb = self.uid_embedding(user_id).squeeze(1)
        iid_emb = self.iid_embedding(item_id).squeeze(1)

        use_fea = torch.cat([user_doc_fea, uid_emb], dim=1)
        item_fea = torch.cat([item_doc_fea, iid_emb], dim=1)

        # predict = [batch size, 1]
        latent = torch.cat([use_fea, item_fea], dim=1)
        predict = self.predict_layer(latent)
        return predict

    def local_attention_cnn(self, word_embs, doc_cnn: torch.nn.Conv2d):
        """
        :param word_embs:   (batch size, review length, word dim)
        :return:            (batch size, kernel num, review length, 1)

        Eq1 - Eq7
        """
        local_att_words = self.word_cnn(word_embs.unsqueeze(1))
        local_word_weight = torch.sigmoid(local_att_words.squeeze(1))
        word_embs = word_embs * local_word_weight
        d_fea = doc_cnn(word_embs.unsqueeze(1))
        return d_fea

    def local_pooling_cnn(self, feature, attention, cnn, fc):
        """
        :param feature:     (batch size, kernel num, review length, 1)
        :param attention:   (batch size, review length)
        :return:            (batch size, latent factors)

        Eq11 - Eq13
        """
        bs, n_filters, doc_len, _ = feature.shape
        feature = feature.permute(0, 3, 2, 1)  # bs * 1 * doc_len * embed
        attention = attention.reshape(bs, 1, doc_len, 1)  # bs * doc
        pools = feature * attention
        pools = self.unfold(pools)
        pools = pools.reshape(bs, 3, n_filters, doc_len)
        pools = pools.sum(dim=1, keepdims=True)  # bs * 1 * n_filters * doc_len
        pools = pools.transpose(2, 3)  # bs * 1 * doc_len * n_filters

        abs_fea = cnn(pools).squeeze(3)  # ? (DOC_LEN-2), 100
        abs_fea = torch.avg_pool1d(abs_fea, abs_fea.size(2))  # ? 100
        abs_fea = torch.relu(fc(abs_fea.squeeze(2)))  # ? 32

        return abs_fea

    def reset_para(self):

        cnns = [self.word_cnn, self.user_doc_cnn, self.item_doc_cnn, self.user_abs_cnn, self.item_abs_cnn]
        for cnn in cnns:
            torch.nn.init.xavier_normal_(cnn.weight)
            torch.nn.init.uniform_(cnn.bias, -0.1, 0.1)

        fcs = [self.user_fc, self.item_fc]
        for fc in fcs:
            torch.nn.init.uniform_(fc.weight, -0.1, 0.1)
            torch.nn.init.constant_(fc.bias, 0.1)

        torch.nn.init.uniform_(self.uid_embedding.weight, -0.1, 0.1)
        torch.nn.init.uniform_(self.iid_embedding.weight, -0.1, 0.1)


class PredictLayer(torch.nn.Module):
    def __init__(self, config: DamlConfig):
        super().__init__()
        self.config = config

        self.mlp_layer = MlpLayer(config.mlp_layers, config.latent_factors * 4, 1, config.dropout, "relu")
        self.lin = torch.nn.Linear(config.latent_factors * 4, 1)

    def forward(self, feature):
        """
        :param feature: (batch size, 1)
        :return:        (batch size, 1)
        """
        lin = self.lin(feature)
        mlp = self.mlp_layer(feature)
        predict = lin + mlp
        return predict


def test():
    config = DamlConfig(review_count=1)
    word_dict = load_word_dict(config.data_set)
    model = DamlModel(config, word_dict.weight)

    review_size = [config.batch_size, config.review_length]
    user_review = torch.randint(config.word_count, review_size).type(torch.long)
    item_review = torch.randint(config.word_count, review_size).type(torch.long)
    user_id = torch.randint(config.user_count, [config.batch_size, 1]).type(torch.long)
    item_id = torch.randint(config.user_count, [config.batch_size, 1]).type(torch.long)
    rating = torch.randint(5, [config.batch_size, 1])

    iter_i = (user_review, item_review, user_id, item_id, rating)
    predict, rating, loss = model.predict_iter_i(iter_i)
    logger.info(predict.shape)
    logger.info(rating.shape)
    logger.info(loss.shape)


if __name__ == '__main__':
    test()
