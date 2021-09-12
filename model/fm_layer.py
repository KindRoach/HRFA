import torch


class FMLayer(torch.nn.Module):
    """
    The implementation of Factorization machine.
    Reference: https://www.kaggle.com/gennadylaptev/factorization-machine-implemented-in-pytorch
    """

    def __init__(self, fm_in: int, fm_k: int):
        super().__init__()
        self.V = torch.nn.Parameter(torch.randn(fm_in, fm_k), requires_grad=True)
        self.lin = torch.nn.Linear(fm_in, 1)

    def forward(self, x):
        """
        :param x:   (batch size, fm_in)
        :return:    (batch size, fm_k)
        """
        s1_square = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)
        s2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)

        out_inter = 0.5 * (s1_square - s2)
        out_lin = self.lin(x)
        out = out_inter + out_lin

        return out


class FMLayerWithBias(torch.nn.Module):
    def __init__(self, fm_in: int, fm_k: int, user_count: int, item_count: int):
        super().__init__()
        self.fm_layer = FMLayer(fm_in, fm_k)
        self.b_user = torch.nn.Parameter(torch.zeros([user_count]), requires_grad=True)
        self.b_item = torch.nn.Parameter(torch.zeros([item_count]), requires_grad=True)

    def forward(self, latent, user_id, item_id):
        """
        :param latent:  (batch size, latent factors * 2)
        :param user_id: (batch size, 1)
        :param item_id: (batch size, 1)
        :return:        (batch size, 1)
        """

        predict = self.fm_layer(latent)
        predict = predict + self.b_user[user_id] + self.b_item[item_id]
        return predict
