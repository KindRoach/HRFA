from data_process.dataset.mrmrp_het_dataset import MrmrpHetDataset
from model.mrmrp import MrmrpConfig, MrmrpModel


class MrmrpHetModel(MrmrpModel):

    def __init__(self, config: MrmrpConfig, word_embedding_weight):
        super().__init__(config, word_embedding_weight)
        self.dataset_class = MrmrpHetDataset
