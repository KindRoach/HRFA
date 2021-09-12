import json
from typing import Optional

from data_process.data_reader.data_enum import DataSetEnum
from tool.path_helper import ROOT_DIR, create_path

BEST_CONFIG_PATH = "out/tune/best_config.json"


###################
# Thread Unsafe ! #
###################

def load_best_config():
    best_config_path = ROOT_DIR.joinpath(BEST_CONFIG_PATH)
    if best_config_path.exists():
        with open(best_config_path, "r", encoding="utf-8") as f:
            best_config = json.loads(f.read())
    else:
        best_config = dict()

    return best_config


def save_config(best_config: dict):
    best_config_path = ROOT_DIR.joinpath(BEST_CONFIG_PATH)
    create_path(best_config_path)
    with open(best_config_path, 'w', encoding='utf-8') as f:
        json.dump(best_config, f, ensure_ascii=False, indent=2)


def update_best_config(model_class: type, data_set: DataSetEnum, config: dict):
    best_config = load_best_config()
    mode_name = model_class.__name__
    if mode_name not in best_config:
        best_config[mode_name] = dict()
    best_config[mode_name][data_set.value] = config
    save_config(best_config)


def read_best_config(model_class: type, data_set: DataSetEnum) -> Optional[dict]:
    best_config = load_best_config()
    try:
        if "B5" in data_set.value:
            key = data_set.value[:-3]
        else:
            key = data_set.value
        return best_config[model_class.__name__][key]
    except KeyError:
        return None
