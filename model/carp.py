from dataclasses import dataclass

import torch

from data_process.dataset.anr_dataset import AnrDataset
from model.base_model import BaseConfig, BaseModel
from tool.log_helper import logger
from tool.word2vec_helper import load_word_dict


@dataclass
class CarpConfig(BaseConfig):
    max_rating: int = 5
    min_rating: int = 1
    kernel_num: int = 50
    kernel_width: int = 3
    aspect_num: int = 5
    iteration: int = 3
    sqr_loss_weight: float = 0.5  # λ in eq.14
    cor_threshold: float = 0.8  # ϵ in eq.13


class CarpModel(BaseModel):
    def __init__(self, config: CarpConfig, word_embedding_weight):
        super().__init__(config)
        self.config = config

        word_embedding = torch.nn.Embedding.from_pretrained(word_embedding_weight, freeze=False, padding_idx=config.pad_id)
        self.user_encoder = CnnEncoder(config, word_embedding)
        self.item_encoder = CnnEncoder(config, word_embedding)
        self.user_view_layer = SelfAttentionLayer(config)
        self.item_asp_layer = SelfAttentionLayer(config)
        self.pos_cap = SentimentCapsule(config)
        self.neg_cap = SentimentCapsule(config)
        self.predict_layer = PredictionLayer(config)

        self.loss_f = torch.nn.MSELoss()
        self.dataset_class = AnrDataset

    def forward(self, user_review, item_review, user_id, item_id):
        """
        :param user_review: (batch size, review length)
        :param item_review: (batch size, review length)
        :param user_id:     (batch size, 1)
        :param item_id:     (batch size, 1)
        :return:            (batch size, 1)
        """
        user_review_emb = self.user_encoder(user_review)
        item_review_emb = self.item_encoder(item_review)

        mask = user_review == self.config.pad_id
        user_view = self.user_view_layer(user_review_emb, mask)

        mask = user_review == self.config.pad_id
        item_asp = self.item_asp_layer(item_review_emb, mask)

        logic = self.logic_unit(user_view, item_asp)
        pos = self.pos_cap(logic)
        neg = self.neg_cap(logic)
        r = self.predict_layer(pos, neg, user_id, item_id)

        stm_loss = self.stm_loss(pos, neg)

        return r, stm_loss

    def logic_unit(self, user_view, item_asp):
        """
        :param user_view:   (batch size, kernel num, aspect num)
        :param item_asp:    (batch size, kernel num, aspect num)
        :return:            (batch size, kernel num * 2, aspect num ** 2)
        """
        g_x_y = []
        for x in range(self.config.aspect_num):
            for y in range(self.config.aspect_num):
                sub = user_view[:, :, x] - item_asp[:, :, y]
                dot = user_view[:, :, x] * item_asp[:, :, y]
                cat = torch.cat([sub, dot], dim=1)
                g_x_y.append(cat)

        g_x_y = torch.stack(g_x_y, dim=2)
        return g_x_y

    def stm_loss(self, pos, neg):
        """
        :param pos: (batch size, kernel num * 2)
        :param neg: (batch size, kernel num * 2)
        :return:    (1)
        """

        norm_p = torch.norm(pos, p=2, dim=1)
        zero = torch.zeros_like(norm_p)
        max_p = torch.max(zero, self.config.cor_threshold - norm_p)

        norm_n = torch.norm(neg, p=2, dim=1)
        zero = torch.zeros_like(norm_p)
        max_n = torch.max(zero, self.config.cor_threshold + norm_n - 1)

        stm_loss = (max_p + max_n).mean()
        return stm_loss

    def predict_iter_i(self, iter_i):
        iter_i = self.move_iter_i_to_device(iter_i)
        args = iter_i[:-1]
        rating = iter_i[-1]
        predict, stm_loss = self(*args)
        sqr_loss = self.config.sqr_loss_weight * self.loss_f(predict, rating)
        stm_loss = (1 - self.config.sqr_loss_weight) * stm_loss
        loss = sqr_loss + stm_loss
        return predict, rating, loss


class CnnEncoder(torch.nn.Module):
    def __init__(self, config: CarpConfig, word_emb):
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
        :param review:  (batch size, review len)
        :return:        (batch size, review len, kernel num)
        """

        # review_emb = [batch size, review_length, word dim]
        review_emb = self.word_emb(review)

        # review_emb = [batch size, word dim, review len]
        review_emb = review_emb.permute(0, 2, 1)

        # cnn_out = [batch size, review length, kernel num]
        outputs = torch.relu(self.word_cnn(review_emb))
        outputs = self.dropout(outputs).permute(0, 2, 1)

        return outputs


class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, config: CarpConfig):
        super().__init__()
        self.config = config

        self.asp_embeddings = torch.nn.Embedding(config.aspect_num, config.kernel_num)
        self.Ws_word = torch.nn.ModuleList()
        self.Ws_asp = torch.nn.ModuleList()
        self.Ws_prj = torch.nn.ModuleList()

        for i in range(config.aspect_num):
            self.Ws_word.append(torch.nn.Linear(config.kernel_num, config.kernel_num, bias=False))
            self.Ws_asp.append(torch.nn.Linear(config.kernel_num, config.kernel_num))
            self.Ws_prj.append(torch.nn.Linear(config.kernel_num, config.kernel_num, bias=False))

        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, review, mask):
        """
        :param review:  (batch size, review len, kernel num)
        :param mask:    (batch size, review len)
        :return:        (batch size, kernel num, aspect num)
        """

        v_u = []
        for x in range(self.config.aspect_num):
            # p_u_x_j = [batch size, review len, kernel num]
            asp_idx = torch.full_like(review[:, :, 0], x, dtype=torch.int64)
            q_u_x = self.asp_embeddings(asp_idx)
            linear = self.Ws_word[x](review) + self.Ws_asp[x](q_u_x)
            s_u_x_j = review * torch.sigmoid(linear)
            p_u_x_j = self.Ws_prj[x](s_u_x_j)
            p_u_x_j = self.dropout(p_u_x_j)

            # attn_u_x_j = [batch size, review len, 1]
            v_u_x = p_u_x_j.mean(dim=1)
            for i in range(self.config.iteration):
                mul = p_u_x_j @ v_u_x.unsqueeze(2)
                mul = mul.masked_fill(mask.unsqueeze(2), -1e10)
                attn_u_x_j = torch.softmax(mul, dim=1)

                v_u_x = p_u_x_j.permute(0, 2, 1) @ attn_u_x_j
                v_u_x = v_u_x.squeeze(2)

            v_u.append(v_u_x)

        # v_u = [batch size, kernel num, aspect num]
        v_u = torch.stack(v_u, dim=2)
        return v_u


class SentimentCapsule(torch.nn.Module):
    def __init__(self, config: CarpConfig):
        super().__init__()
        self.config = config
        self.W_cap = torch.nn.Linear(config.aspect_num ** 2, config.aspect_num ** 2, bias=False)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, logic):
        """
        :param logic:   (batch size, kernel num * 2, aspect num ** 2)
        :return:        (batch size, kernel num * 2)
        """

        # t_s_x_y = [batch size, kernel num * 2, aspect num ** 2]
        t_s_x_y = self.dropout(self.W_cap(logic))

        # b_s_x_y = [batch size, aspect num ** 2, 1]
        b_s_x_y = torch.zeros([t_s_x_y.shape[0], t_s_x_y.shape[2], 1], device=self.config.device)
        o_s_u_i = None

        for i in range(self.config.iteration):
            # s_s_u_i = [batch size, kernel num * 2]
            c_s_x_y = torch.softmax(b_s_x_y, dim=1)
            s_s_u_i = (t_s_x_y @ c_s_x_y).squeeze(2)

            # squashing
            # o_s_u_i = [batch size, kernel num * 2]
            norm = torch.norm(s_s_u_i, p=2, dim=1)
            squ_weight = (norm ** 2 / (1 + norm ** 2)).unsqueeze(1)
            o_s_u_i = squ_weight * s_s_u_i / norm.unsqueeze(1)

            b_update = (o_s_u_i.unsqueeze(1) @ t_s_x_y).permute(0, 2, 1)
            b_s_x_y = b_s_x_y + b_update

        return o_s_u_i


class PredictionLayer(torch.nn.Module):
    def __init__(self, config: CarpConfig):
        super().__init__()
        self.config = config

        self.pos_high = HighwayLayer(config)
        self.neg_high = HighwayLayer(config)
        self.bu = torch.nn.Embedding(self.config.user_count, 1)
        self.bi = torch.nn.Embedding(self.config.item_count, 1)

    def forward(self, pos, neg, user_id, item_id):
        """
        :param pos:     (batch size, kernel num * 2)
        :param neg:     (batch size, kernel num * 2)
        :param user_id: (batch size, 1)
        :param item_id: (batch size, 1)
        :return:        (batch size, 1)
        """

        pos_r = self.pos_high(pos)
        pos_r = pos_r * torch.norm(pos, p=2, dim=1).unsqueeze(1)

        neg_r = self.pos_high(neg)
        neg_r = neg_r * torch.norm(neg, p=2, dim=1).unsqueeze(1)

        diff = pos_r - neg_r
        fc = self.config.min_rating + (self.config.max_rating - 1) / (1 + torch.exp(-diff))
        r = fc + self.bu(user_id).squeeze(1) + self.bi(item_id).squeeze(1)

        return r


class HighwayLayer(torch.nn.Module):
    def __init__(self, config: CarpConfig):
        super().__init__()
        self.config = config

        self.H_1 = torch.nn.Linear(self.config.kernel_num * 2, 1)
        self.H_2 = torch.nn.Linear(self.config.kernel_num * 2, 1)
        self.W_rate = torch.nn.Linear(self.config.kernel_num * 2, 1)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, low_way):
        """
        :param low_way: (batch size, kernel num * 2)
        :return:        (batch size, 1)
        """

        gate = torch.sigmoid(self.H_1(low_way))
        tanh = torch.tanh(self.H_2(low_way))
        h_s_u_i = gate * low_way + (1 - gate) * tanh
        r = self.dropout(self.W_rate(h_s_u_i))
        return r


def test():
    config = CarpConfig(review_count=1)
    word_dict = load_word_dict(config.data_set)
    model = CarpModel(config, word_dict.weight)

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
