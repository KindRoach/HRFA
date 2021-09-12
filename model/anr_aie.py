import torch

from model.anr_config import AnrConfig


class AnrAie(torch.nn.Module):
    """
    Aspect Importance Estimation (AIE)
    """

    def __init__(self, config: AnrConfig):
        super().__init__()
        self.config = config

        # Matrix for Interaction between User Aspect-level Representations & Item Aspect-level Representations
        # This is a learnable (h1 x h1) matrix, i.e. User Aspects - Rows, Item Aspects - Columns
        self.W_a = torch.nn.Parameter(torch.Tensor(self.config.h1, self.config.h1), requires_grad=True)

        # User "Projection": A (h2 x h1) weight matrix, and a (h2 x 1) vector
        self.W_u = torch.nn.Parameter(torch.Tensor(self.config.h2, self.config.h1), requires_grad=True)
        self.w_hu = torch.nn.Parameter(torch.Tensor(self.config.h2, 1), requires_grad=True)

        # Item "Projection": A (h2 x h1) weight matrix, and a (h2 x 1) vector
        self.W_i = torch.nn.Parameter(torch.Tensor(self.config.h2, self.config.h1), requires_grad=True)
        self.w_hi = torch.nn.Parameter(torch.Tensor(self.config.h2, 1), requires_grad=True)

        # Initialize all weights using random uniform distribution from [-0.01, 0.01]
        self.W_a.data.uniform_(-0.01, 0.01)
        self.W_u.data.uniform_(-0.01, 0.01)
        self.w_hu.data.uniform_(-0.01, 0.01)
        self.W_i.data.uniform_(-0.01, 0.01)
        self.w_hi.data.uniform_(-0.01, 0.01)

    def forward(self, userAspRep, itemAspRep):
        """
        :param userAspRep:  (batch size, num aspects, h1)
        :param itemAspRep:  (batch size, num aspects, h1)
        :return:            (batch size, num aspects)
        """

        userAspRepTrans = torch.transpose(userAspRep, 1, 2)
        itemAspRepTrans = torch.transpose(itemAspRep, 1, 2)

        # Affinity Matrix (User Aspects x Item Aspects), i.e. User Aspects - Rows, Item Aspects - Columns
        affinityMatrix = torch.matmul(userAspRep, self.W_a)
        affinityMatrix = torch.matmul(affinityMatrix, itemAspRepTrans)

        # Non-Linearity: ReLU
        affinityMatrix = torch.relu(affinityMatrix)

        # -------------------  User Importance (over Aspects) -------------------
        H_u_1 = torch.matmul(self.W_u, userAspRepTrans)
        H_u_2 = torch.matmul(self.W_i, itemAspRepTrans)
        H_u_2 = torch.matmul(H_u_2, torch.transpose(affinityMatrix, 1, 2))
        H_u = H_u_1 + H_u_2

        # Non-Linearity: ReLU
        H_u = torch.relu(H_u)

        # User Aspect-level Importance
        userAspImpt = torch.matmul(torch.transpose(self.w_hu, 0, 1), H_u)

        # User Aspect-level Importance: (bsz x 1 x num_aspects) -> (bsz x num_aspects x 1)
        userAspImpt = torch.transpose(userAspImpt, 1, 2)
        userAspImpt = torch.softmax(userAspImpt, dim=1)

        # User Aspect-level Importance: (bsz x num_aspects x 1) -> (bsz x num_aspects)
        userAspImpt = torch.squeeze(userAspImpt, 2)

        # ------------------- Item Importance (over Aspects) -------------------
        H_i_1 = torch.matmul(self.W_i, itemAspRepTrans)
        H_i_2 = torch.matmul(self.W_u, userAspRepTrans)
        H_i_2 = torch.matmul(H_i_2, affinityMatrix)
        H_i = H_i_1 + H_i_2

        # Non-Linearity: ReLU
        H_i = torch.relu(H_i)

        # Item Aspect-level Importance
        itemAspImpt = torch.matmul(torch.transpose(self.w_hi, 0, 1), H_i)

        # Item Aspect-level Importance: (bsz x 1 x num_aspects) -> (bsz x num_aspects x 1)
        itemAspImpt = torch.transpose(itemAspImpt, 1, 2)
        itemAspImpt = torch.softmax(itemAspImpt, dim=1)

        # Item Aspect-level Importance: (bsz x num_aspects x 1) -> (bsz x num_aspects)
        itemAspImpt = torch.squeeze(itemAspImpt, 2)

        return userAspImpt, itemAspImpt
