import time

import pandas
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_process.data_reader.data_enum import DataTypeEnum
from model.base_model import BaseModel
from tool.exp_result_helper import report_eval
from tool.log_helper import logger
from tool.model_helper import load_model
from tool.path_helper import ROOT_DIR, create_path


def load_test_model(model_save_name, device):
    model = load_model(model_save_name)
    config = model.config

    if device is not None:
        config.device = device

    model.to(config.device)
    return model


def test_model(model_save_name: str, device: str = None):
    model: BaseModel = load_test_model(model_save_name, device)
    config = model.config

    test_dataset = model.create_dataset(DataTypeEnum.Test)
    test_data_loader = DataLoader(test_dataset, batch_size=model.config.batch_size * 3, shuffle=False)

    loss_f = torch.nn.MSELoss()

    model.eval()
    model.record_mid_var = True
    predicts = []
    actuals = []
    with torch.no_grad():
        for iter_i in tqdm(test_data_loader, desc="Evaluating"):
            predict, actual, _ = model.predict_iter_i(iter_i)
            predicts.append(predict)
            actuals.append(actual)

    predicts = torch.cat(predicts)
    actuals = torch.cat(actuals)
    loss = loss_f(predicts, actuals).item()
    logger.info(f"{model_save_name} MSE = {loss}")
    report_eval(model_save_name, config.data_set.value, {"MSE": loss})

    logger.info("Writing result...")
    result_path = ROOT_DIR.joinpath(f"out/predict/{model_save_name}.csv")
    create_path(result_path)
    with open(result_path, "w", encoding="utf-8") as f:
        mse = (actuals - predicts) ** 2
        for i in mse:
            f.write(f"{i.item()}\n")
    return loss


def main():
    torch.set_num_threads(1)
    eval_result = dict()
    for name in []:
        loss = test_model(name, "cuda:0")
        model_class = name.split("_")[0]
        data_set = name.split("_")[1]
        if model_class not in eval_result:
            eval_result[model_class] = dict()
        eval_result[model_class][data_set] = loss

    eval_result = pandas.DataFrame(eval_result)
    eval_path = ROOT_DIR.joinpath(f"out/eval/{time.strftime('%Y%m%d%H%M%S', time.localtime())}.csv")
    create_path(eval_path)
    eval_result.to_csv(eval_path, float_format='%.4f')


if __name__ == '__main__':
    main()
