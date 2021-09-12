import torch

from tool.log_helper import logger

ACTIVATION_FUNC_MAP = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
}


class MlpLayer(torch.nn.Module):
    def __init__(self, layers: int, input_dim: int, output_dim: int, dropout: float = 0, activation_func: str = "tanh"):
        super().__init__()

        self.layers = layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = torch.nn.Dropout(dropout)
        self.activation_func = ACTIVATION_FUNC_MAP[activation_func]()

        dim_diff = (input_dim - output_dim) // (1 + layers)
        mlp_dims = [input_dim - dim_diff * i for i in range(1 + layers)]
        mlp_dims.append(output_dim)
        self.mlp_dims = mlp_dims

        models = []
        for in_size, out_size in zip(mlp_dims[:-1], mlp_dims[1:]):
            models.append(torch.nn.Linear(in_size, out_size))
            models.append(self.activation_func)
            models.append(self.dropout)

        self.mlp = torch.nn.Sequential(*models)

    def forward(self, input_vec):
        """
        :param input_vec:   (batch size, input dim)
        :return:            (batch size, 1)
        """

        out = self.mlp(input_vec)
        return out


def test():
    mlp = MlpLayer(2, 16, 8)
    input = torch.rand([64, 16])
    output = mlp(input)
    logger.info(mlp.mlp_dims)
    logger.info(output.shape)


if __name__ == '__main__':
    test()
