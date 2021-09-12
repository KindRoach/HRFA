import torch

from model.anr_config import AnrConfig


class AnrRatingPred(torch.nn.Module):
    """
    Aspect-Based Rating Predictor, using Aspect-based Representations & the estimated Aspect Importance
    """

    def __init__(self, config: AnrConfig):
        super().__init__()
        self.config = config

        # Dropout for the User & Item Aspect-Based Representations
        self.userAspRepDropout = torch.nn.Dropout(p=self.config.dropout)
        self.itemAspRepDropout = torch.nn.Dropout(p=self.config.dropout)

        # Global Offset/Bias (Trainable)
        self.globalOffset = torch.nn.Parameter(torch.Tensor(1), requires_grad=True)

        # User Offset/Bias & Item Offset/Bias
        self.uid_userOffset = torch.nn.Embedding(self.config.user_count, 1)
        self.iid_itemOffset = torch.nn.Embedding(self.config.item_count, 1)

        # Initialize Global Bias with 0
        self.globalOffset.data.fill_(0)

        # Initialize All User/Item Offset/Bias with 0
        self.uid_userOffset.weight.data.fill_(0)
        self.iid_itemOffset.weight.data.fill_(0)

    def forward(self, userAspRep, itemAspRep, userAspImpt, itemAspImpt, user_id, item_id):
        """
        :param userAspRep:  (batch size, num aspects, h1)
        :param itemAspRep:  (batch size, num aspects, h1)
        :param userAspImpt: (batch size, num aspects)
        :param itemAspImpt: (batch size, num aspects)
        :param user_id:     (batch size, 1)
        :param item_id:     (batch size, 1)
        :return:            (batch size, 1)
        """

        # User & Item Bias
        batch_userOffset = self.uid_userOffset(user_id).squeeze(1)
        batch_itemOffset = self.iid_itemOffset(item_id).squeeze(1)

        userAspRep = self.userAspRepDropout(userAspRep)
        itemAspRep = self.itemAspRepDropout(itemAspRep)

        lstAspRating = []

        # (bsz x num_aspects x h1) -> (num_aspects x bsz x h1)
        userAspRep = torch.transpose(userAspRep, 0, 1)
        itemAspRep = torch.transpose(itemAspRep, 0, 1)

        for k in range(self.config.num_aspects):
            user = torch.unsqueeze(userAspRep[k], 1)
            item = torch.unsqueeze(itemAspRep[k], 2)
            aspRating = torch.matmul(user, item)
            aspRating = torch.squeeze(aspRating, 2)
            lstAspRating.append(aspRating)

        # List of (bsz x 1) -> (bsz x num_aspects)
        rating_pred = torch.cat(lstAspRating, dim=1)

        # Multiply Each Aspect-Level (Predicted) Rating with the Corresponding User-Aspect Importance & Item-Aspect Importance
        rating_pred = userAspImpt * itemAspImpt * rating_pred

        # Sum over all Aspects
        rating_pred = torch.sum(rating_pred, dim=1, keepdim=True)

        # Include User Bias & Item Bias
        rating_pred = rating_pred + batch_userOffset + batch_itemOffset

        # Include Global Bias
        rating_pred = rating_pred + self.globalOffset

        return rating_pred
