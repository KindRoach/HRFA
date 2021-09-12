import torch

from data_process.dataset.anr_dataset import AnrDataset
from model.anr_aie import AnrAie
from model.anr_arl import AmrArl
from model.anr_config import AnrConfig
from model.anr_rating_pred import AnrRatingPred
from model.base_model import BaseModel
from tool.log_helper import logger
from tool.word2vec_helper import load_word_dict


class AnrModel(BaseModel):
    """
    This is the complete Aspect-based Neural Recommender (ANR),
    with ARL and AIE as its main components.
    """

    def __init__(self, config: AnrConfig, word_embedding_weight):
        super().__init__(config)

        # User Documents & Item Documents (Input)
        self.uid_userDoc = torch.nn.Embedding(config.user_count, config.review_length)
        self.iid_itemDoc = torch.nn.Embedding(config.item_count, config.review_length)
        self.wid_wEmbed = torch.nn.Embedding.from_pretrained(word_embedding_weight, freeze=False, padding_idx=config.pad_id)

        # Aspect Representation Learning - Single Aspect-based Attention Network (Shared between User & Item)
        self.shared_ANR_ARL = AmrArl(config)

        # Aspect-Based Co-Attention (Parallel Co-Attention, using the Affinity Matrix as a Feature) --- Aspect Importance Estimation
        self.ANR_AIE = AnrAie(config)

        # Aspect-Based Rating Predictor based on the estimated Aspect-Level Importance
        self.ANR_RatingPred = AnrRatingPred(config)

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

        # Embedding Layer
        batch_userDocEmbed = self.wid_wEmbed(user_review)
        batch_itemDocEmbed = self.wid_wEmbed(item_review)

        # Aspect-based Representation Learning for User
        userAspAttn, userAspDoc = self.shared_ANR_ARL(batch_userDocEmbed)

        # Aspect-based Representation Learning for Item
        itemAspAttn, itemAspDoc = self.shared_ANR_ARL(batch_itemDocEmbed)

        # Aspect-based Co-Attention --- Aspect Importance Estimation
        userCoAttn, itemCoAttn = self.ANR_AIE(userAspDoc, itemAspDoc)

        # Aspect-Based Rating Predictor based on the estimated Aspect-Level Importance
        rating_pred = self.ANR_RatingPred(userAspDoc, itemAspDoc, userCoAttn, itemCoAttn, user_id, item_id)

        return rating_pred


def test():
    config = AnrConfig(review_count=1)
    word_dict = load_word_dict(config.data_set)
    model = AnrModel(config, word_dict.weight)

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
