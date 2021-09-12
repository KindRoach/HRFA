import logging

import ray
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.progress_reporter import TuneReporterBase

from data_process.data_reader.data_enum import DataSetEnum, DataTypeEnum, REVIEW_LENGTH, REVIEW_COUNT
from data_process.dataset.base_dataset import BaseDataset
from model.anr import AnrModel
from model.anr_config import AnrConfig
from model.base_model import BaseConfig
from model.carp import CarpModel, CarpConfig
from model.daml import DamlModel, DamlConfig
from model.deepconn import DeepConnConfig, DeepConnModel
from model.hrfa import HrfaConfig, HrfaModel
from model.hrfa_bias import HrfaBiasModel
from model.hrfa_noadj import HrfaNoadjModel
from model.hrfa_self import HrfaSelfModel
from model.hrfa_sim import HrfaSimModel
from model.hrfa_three_way import HrfaThreeWayModel
from model.mrmrp import MrmrpModel, MrmrpConfig
from model.mrmrp_het import MrmrpHetModel
from model.nrpa import NrpaConfig, NrpaModel
from model.parl import ParlConfig, ParlModel
from model.parl_het import ParlHetModel
from tool.best_config_helper import read_best_config
from tool.log_helper import logger
from tool.model_helper import NAME_TO_MODEL
from tool.path_helper import ROOT_DIR
from tool.train_helper import TrainHelper, TrainModeEnum
from tool.word2vec_helper import load_word_dict


def train_with_config(model_class, config, data_set: DataSetEnum, train_mode: TrainModeEnum, train_tag: str = ""):
    word_dict = load_word_dict(data_set)
    base_dataset = BaseDataset(data_set, DataTypeEnum.Train)
    update_common_config(model_class, config, base_dataset, word_dict, train_mode)
    model = model_class(config, word_dict.weight)
    model.to(config.device)
    train_helper = TrainHelper(model, train_tag, train_mode)
    train_helper.train_model()


def update_common_config(model_class, config: BaseConfig, base_dataset, word_dict, train_mode):
    # read best config from tune result
    if not train_mode == TrainModeEnum.Tune:
        best_config = read_best_config(model_class, base_dataset.data_set)
        config.batch_size = best_config["bt"]
        config.learning_rate = best_config["lr"]
        config.l2_regularization = best_config["reg"]

    # Update Common Config
    config.data_set = base_dataset.data_set
    config.steps_num = 100000
    config.learning_rate_decay = 0.99
    if not torch.cuda.is_available():
        config.device = "cpu"
    config.word_dim = word_dict.weight.shape[1]
    config.word_count = word_dict.weight.shape[0]
    config.user_count = base_dataset.user_count
    config.item_count = base_dataset.item_count
    config.avg_rating = base_dataset.avg_rating


def main_train(model_class: type, data_set: DataSetEnum, train_mode=TrainModeEnum.Single, train_tag: str = ""):
    hrfa_config = HrfaConfig(
        device="cuda:0",
        review_length=REVIEW_LENGTH,
        review_count=REVIEW_COUNT,
        kernel_width=3,
        kernel_num=40,
        latent_factors=30,
        id_dim=32,
        fm_k=8
    )

    config_map = {
        DeepConnModel: DeepConnConfig(
            device="cuda:0",
            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,
            kernel_width=3,
            kernel_num=100,
            latent_factors=50,
            fm_k=8
        ),

        ParlModel: ParlConfig(
            device="cuda:0",
            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,
            aux_loss_weight=0.03,
            kernel_width=3,
            kernel_num=100,
            latent_factors=50,
            fm_k=8
        ),

        ParlHetModel: ParlConfig(
            device="cuda:0",
            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,
            aux_loss_weight=0.03,
            kernel_width=3,
            kernel_num=100,
            latent_factors=50,
            fm_k=8
        ),

        NrpaModel: NrpaConfig(
            device="cuda:0",
            review_length=REVIEW_LENGTH,
            review_count=REVIEW_COUNT,
            kernel_width=3,
            kernel_num=80,
            id_dim=32,
            fm_k=8
        ),

        MrmrpModel: MrmrpConfig(
            device="cuda:0",
            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,
            kernel_width=5,
            kernel_num=128,
            latent_factors=50,
            mlp_layers=3
        ),

        MrmrpHetModel: MrmrpConfig(
            device="cuda:0",
            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,
            kernel_width=5,
            kernel_num=128,
            latent_factors=50,
            mlp_layers=1
        ),

        AnrModel: AnrConfig(
            device="cuda:0",
            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,
            h1=10,
            h2=50,
            ctx_win_size=3,
            num_aspects=5
        ),

        DamlModel: DamlConfig(
            device="cuda:0",
            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,
            kernel_width=3,
            kernel_num=100,
            latent_factors=32,
            mlp_layers=1,
        ),

        CarpModel: CarpConfig(
            device="cuda:0",
            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,
            kernel_width=3,
            kernel_num=100,
            max_rating=5,
            min_rating=1,
            aspect_num=5,
            iteration=3,
            sqr_loss_weight=0.5,
            cor_threshold=0.8,
        ),

        HrfaModel: hrfa_config,
        HrfaSelfModel: hrfa_config,
        HrfaSimModel: hrfa_config,
        HrfaNoadjModel: hrfa_config,
        HrfaBiasModel: hrfa_config,
        HrfaThreeWayModel: hrfa_config,
    }

    train_with_config(model_class, config_map[model_class], data_set, train_mode, train_tag)


def main():
    data_func_space = {
        "data_set": tune.grid_search([
            DataSetEnum.Appliances,
            DataSetEnum.Luxury_Beauty,
            DataSetEnum.Prime_Pantry,
            DataSetEnum.Digital_Music,
            DataSetEnum.Yelp
        ]),

        "_model": tune.grid_search([
            DeepConnModel.__name__,
            NrpaModel.__name__,
            ParlModel.__name__,
            ParlHetModel.__name__,
            MrmrpModel.__name__,
            MrmrpHetModel.__name__,
            AnrModel.__name__,
            DamlModel.__name__,
            CarpModel.__name__,
            HrfaModel.__name__,
        ])
    }

    gpus_num = torch.cuda.device_count()
    logger.info(f"Using {gpus_num} GPUs")
    ray.init(logging_level=logging.ERROR, num_gpus=gpus_num)

    def tune_func(config, checkpoint_dir=None):
        main_train(NAME_TO_MODEL[config["_model"]], config["data_set"], TrainModeEnum.Multi)

    reporter_clos = TuneReporterBase.DEFAULT_COLUMNS.copy()
    reporter_clos["dev_metric"] = "mse"
    reporter = CLIReporter(metric_columns=reporter_clos)

    tune.run(
        tune_func,
        name="Main_Train",
        verbose=1,
        sync_to_driver=False,
        local_dir=ROOT_DIR.joinpath("out/tune"),
        resources_per_trial={"cpu": 1, "gpu": 1},
        progress_reporter=reporter,
        config=data_func_space)


if __name__ == '__main__':
    main_train(HrfaModel, DataSetEnum.Prime_Pantry)
