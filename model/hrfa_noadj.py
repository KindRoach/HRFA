from data_process.dataset.hrfa_noadj_dataset import HrfaNoadjDataset
from model.hrfa import HrfaConfig
from model.hrfa_sim import HrfaSimModel


class HrfaNoadjModel(HrfaSimModel):
    def __init__(self, config: HrfaConfig, word_embedding_weight):
        super().__init__(config, word_embedding_weight)
        self.dataset_class = HrfaNoadjDataset
