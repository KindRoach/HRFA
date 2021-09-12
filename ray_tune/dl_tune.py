import logging
import time

import ray
import torch
from ray import tune

from data_process.data_reader.data_enum import REVIEW_LENGTH, REVIEW_COUNT, DataSetEnum
from model.anr import AnrModel
from model.anr_config import AnrConfig
from model.carp import CarpModel, CarpConfig
from model.daml import DamlModel, DamlConfig
from model.deepconn import DeepConnConfig, DeepConnModel
from model.hrfa import HrfaConfig, HrfaModel
from model.hrfa_self import HrfaSelfModel
from model.mrmrp import MrmrpConfig, MrmrpModel
from model.nrpa import NrpaConfig, NrpaModel
from model.parl import ParlConfig, ParlModel
from tool.best_config_helper import update_best_config, read_best_config
from tool.log_helper import logger, add_log_file, remove_log_file
from tool.train_helper import TrainModeEnum
from tool.tune_hepler import run_tune
from train import train_with_config

DEFAULT_SEARCH_SPACE = {
    "bt": tune.grid_search([8, 16, 32, 64]),
    "lr": tune.grid_search([1e-6, 1e-5, 1e-4, 1e-3, 0.01]),
    "reg": tune.grid_search([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01]),
}

GPU_NUM_MAP = {
    DeepConnModel: 0.2,
    ParlModel: 0.3,
    NrpaModel: 0.3,
    MrmrpModel: 0.25,
    AnrModel: 0.3,
    DamlModel: 0.3,
    CarpModel: 0.5,
    HrfaModel: 0.5,
    HrfaSelfModel: 0.5,
}


def adjust_search_space(model_class: type, data_set: DataSetEnum):
    search_space = DEFAULT_SEARCH_SPACE.copy()
    config = read_best_config(model_class, data_set)

    if config is None:
        return search_space

    base_lr = config["lr"]
    search_space["lr"] = tune.grid_search(
        [base_lr / 5, base_lr / 2, base_lr, base_lr * 2, base_lr * 5]
    )

    base_reg = config["reg"]
    search_space["reg"] = tune.grid_search(
        [base_reg / 5, base_reg / 2, base_reg, base_reg * 2, base_reg * 5]
    )

    return search_space


def get_deepconn_func(data_set):
    def train_deepconn(config, checkpoint_dir=None):
        model_config = DeepConnConfig(
            batch_size=config["bt"],
            learning_rate=config["lr"],
            device="cuda:0",
            l2_regularization=config["reg"],

            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,

            kernel_width=3,
            kernel_num=100,
            latent_factors=50,
            fm_k=8
        )
        train_with_config(DeepConnModel, model_config, data_set, TrainModeEnum.Tune)

    return train_deepconn


def get_parl_func(data_set):
    def train_parl(config, checkpoint_dir=None):
        model_config = ParlConfig(
            batch_size=config["bt"],
            learning_rate=config["lr"],
            device="cuda:0",
            l2_regularization=config["reg"],

            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,

            kernel_width=3,
            kernel_num=100,
            latent_factors=50,
            fm_k=8
        )
        train_with_config(ParlModel, model_config, data_set, TrainModeEnum.Tune)

    return train_parl


def get_nrpa_func(data_set):
    def train_nrpa(config, checkpoint_dir=None):
        model_config = NrpaConfig(
            batch_size=config["bt"],
            learning_rate=config["lr"],
            device="cuda:0",
            l2_regularization=config["reg"],

            review_length=REVIEW_LENGTH,
            review_count=REVIEW_COUNT,

            kernel_width=3,
            kernel_num=100,
            id_dim=32
        )
        train_with_config(NrpaModel, model_config, data_set, TrainModeEnum.Tune)

    return train_nrpa


def get_mrmrp_func(data_set):
    def train_mrmrp(config, checkpoint_dir=None):
        model_config = MrmrpConfig(
            batch_size=config["bt"],
            learning_rate=config["lr"],
            device="cuda:0",
            l2_regularization=config["reg"],

            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,

            kernel_width=5,
            kernel_num=128,
            latent_factors=50,
            mlp_layers=3
        )
        train_with_config(MrmrpModel, model_config, data_set, TrainModeEnum.Tune)

    return train_mrmrp


def get_anr_func(data_set):
    def train_anr(config, checkpoint_dir=None):
        model_config = AnrConfig(
            batch_size=config["bt"],
            learning_rate=config["lr"],
            device="cuda:0",
            l2_regularization=config["reg"],

            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,

            h1=10,
            h2=50,
            ctx_win_size=3,
            num_aspects=5
        )
        train_with_config(AnrModel, model_config, data_set, TrainModeEnum.Tune)

    return train_anr


def get_daml_func(data_set):
    def train_dmal(config, checkpoint_dir=None):
        model_config = DamlConfig(
            batch_size=config["bt"],
            learning_rate=config["lr"],
            device="cuda:0",
            l2_regularization=config["reg"],

            review_length=REVIEW_LENGTH * REVIEW_COUNT,
            review_count=1,

            kernel_width=3,
            kernel_num=100,
            latent_factors=32,
            mlp_layers=1
        )
        train_with_config(DamlModel, model_config, data_set, TrainModeEnum.Tune)

    return train_dmal


def get_carp_func(data_set):
    def train_carp(config, checkpoint_dir=None):
        model_config = CarpConfig(
            batch_size=config["bt"],
            learning_rate=config["lr"],
            device="cuda:0",
            l2_regularization=config["reg"],

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
        )
        train_with_config(CarpModel, model_config, data_set, TrainModeEnum.Tune)

    return train_carp


def get_hrfa_func(data_set):
    def train_hrfa(config, checkpoint_dir=None):
        model_config = HrfaConfig(
            batch_size=config["bt"],
            learning_rate=config["lr"],
            device="cuda:0",
            l2_regularization=config["reg"],

            review_length=REVIEW_LENGTH,
            review_count=REVIEW_COUNT,

            kernel_width=3,
            kernel_num=40,
            latent_factors=30,
            id_dim=32
        )
        train_with_config(HrfaModel, model_config, data_set, TrainModeEnum.Tune)

    return train_hrfa


def main():
    gpus_num = torch.cuda.device_count()
    add_log_file(logger, f"tune/dl_tune_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.log")
    logger.info(f"Using {gpus_num} GPUs")
    ray.init(logging_level=logging.ERROR, num_gpus=gpus_num)
    for data_set in [
        DataSetEnum.Appliances,
        DataSetEnum.Luxury_Beauty,
        DataSetEnum.Prime_Pantry,
        DataSetEnum.Digital_Music,
        DataSetEnum.Yelp
    ]:
        logger.info(f"---------------------- Tuning on {data_set} ----------------------")
        for model_class, train_func in [
            (DeepConnModel, get_deepconn_func(data_set)),
            (ParlModel, get_parl_func(data_set)),
            (NrpaModel, get_nrpa_func(data_set)),
            (MrmrpModel, get_mrmrp_func(data_set)),
            (AnrModel, get_anr_func(data_set)),
            (DamlModel, get_daml_func(data_set)),
            (CarpModel, get_carp_func(data_set)),
            (HrfaModel, get_hrfa_func(data_set)),
        ]:
            # search_space = adjust_search_space(model_class, data_set)
            # search_mode = "FUll" if search_space == DEFAULT_SEARCH_SPACE else "Detail"
            # exp_name = f"{model_class.__name__}_{data_set}_{search_mode}"
            exp_name = f"{model_class.__name__}_{data_set}"
            best_mse, best_config = run_tune(
                train_func=train_func,
                exp_name=exp_name,
                resources_map={"cpu": 1, "gpu": GPU_NUM_MAP[model_class]},
                search_space=DEFAULT_SEARCH_SPACE,
                metric_name="dev_metric")
            update_best_config(model_class, data_set, best_config)
    remove_log_file(logger)


if __name__ == '__main__':
    main()
