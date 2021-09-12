import logging

import ray
from ray import tune
from surprise import NMF

from data_process.data_reader.data_enum import DataSetEnum
from surprise_method import load_data, run_exp
from tool.best_config_helper import update_best_config
from tool.log_helper import logger
from tool.tune_hepler import run_tune

SEARCH_SPACE_MAP = {
    "lr": tune.grid_search([1e-6, 1e-5, 1e-4, 1e-3, 0.01]),
    "reg_p": tune.grid_search([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01]),
    "reg_b": tune.grid_search([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01]),
}


def get_nmf_func(data_set):
    def train_nmf(config, checkpoint_dir=None):
        train_data, dev_data, _ = load_data(data_set)
        algo = NMF(n_factors=50,
                   n_epochs=10,
                   lr_bu=config["lr"], lr_bi=config["lr"],
                   reg_pu=config["reg_p"], reg_qi=config["reg_p"],
                   reg_bu=config["reg_b"], reg_bi=config["reg_b"])
        tune.report(mse=run_exp(algo, train_data, dev_data), done=True)

    return train_nmf


def main():
    ray.init(logging_level=logging.ERROR)
    for data_set in [
        DataSetEnum.Amazon_Fashion,
        DataSetEnum.Appliances,
        DataSetEnum.Digital_Music,
        DataSetEnum.Luxury_Beauty,
        DataSetEnum.Prime_Pantry,
        DataSetEnum.Beer_Advocate
    ]:
        logger.info(f"---------------------- Tuning on {data_set} ----------------------")
        for model_class, train_func in [
            (NMF, get_nmf_func(data_set))
        ]:
            best_mse, best_config = run_tune(
                train_func=train_func,
                exp_name=f"{model_class.__name__}_{data_set}",
                resources_map={"cpu": 1, "gpu": 0},
                search_space=SEARCH_SPACE_MAP,
                metric_name="mse")
            update_best_config(model_class, data_set, best_config)
            logger.info(f"{model_class} get min mse {best_mse} at {best_config}")


if __name__ == '__main__':
    main()
