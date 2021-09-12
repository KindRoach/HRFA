import warnings

import torch

from model.anr import AnrModel
from model.carp import CarpModel
from model.daml import DamlModel
from model.deepconn import DeepConnModel
from model.hrfa import HrfaModel
from model.hrfa_bias import HrfaBiasModel
from model.hrfa_noadj import HrfaNoadjModel
from model.hrfa_self import HrfaSelfModel
from model.hrfa_sim import HrfaSimModel
from model.hrfa_three_way import HrfaThreeWayModel
from model.mrmrp import MrmrpModel
from model.mrmrp_het import MrmrpHetModel
from model.nrpa import NrpaModel
from model.parl import ParlModel
from model.parl_het import ParlHetModel
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR, create_path
from tool.word2vec_helper import load_word_dict

NAME_TO_MODEL = {
    DeepConnModel.__name__: DeepConnModel,
    NrpaModel.__name__: NrpaModel,
    ParlModel.__name__: ParlModel,
    ParlHetModel.__name__: ParlHetModel,
    MrmrpModel.__name__: MrmrpModel,
    MrmrpHetModel.__name__: MrmrpHetModel,
    AnrModel.__name__: AnrModel,
    DamlModel.__name__: DamlModel,
    CarpModel.__name__: CarpModel,
    HrfaModel.__name__: HrfaModel,
    HrfaSelfModel.__name__: HrfaSelfModel,
    HrfaSimModel.__name__: HrfaSimModel,
    HrfaNoadjModel.__name__: HrfaNoadjModel,
    HrfaBiasModel.__name__: HrfaBiasModel,
    HrfaThreeWayModel.__name__: HrfaThreeWayModel
}

MODEL_PATH = "out/checkpoints/%s.pt"


def save_model(model: torch.nn.Module, model_save_name: str):
    path = ROOT_DIR.joinpath(MODEL_PATH % model_save_name)
    create_path(path)
    torch.save(model, path)
    logger.info(f"model saved: {path}")


def load_model(model_save_name: str):
    path = ROOT_DIR.joinpath(MODEL_PATH % model_save_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # load model to cpu as default.
        model = torch.load(path, map_location=torch.device('cpu'))
    return model


def model_update(model_save_name: str):
    old_model = load_model(model_save_name)
    word_dict = load_word_dict(old_model.config.data_set)
    new_model = old_model.__class__(old_model.config, word_dict.weight)
    new_model.load_state_dict(old_model.state_dict())
    save_model(new_model, model_save_name)


def model_update_config(model_save_name: str):
    old_model = load_model(model_save_name)
    word_dict = load_word_dict(old_model.config.data_set)
    new_model = old_model.__class__(old_model.config, word_dict.weight)
    new_model.load_state_dict(old_model.state_dict())
    save_model(new_model, model_save_name)
