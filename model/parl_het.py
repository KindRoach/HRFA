from data_process.dataset.parl_het_dataset import ParlHetDataset
from model.parl import ParlModel, ParlConfig


class ParlHetModel(ParlModel):

    def __init__(self, config: ParlConfig, word_embedding_weight):
        super().__init__(config, word_embedding_weight)
        self.dataset_class = ParlHetDataset
