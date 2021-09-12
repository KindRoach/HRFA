from dataclasses import dataclass

from model.base_model import BaseConfig


@dataclass
class AnrConfig(BaseConfig):
    h1: int = 10
    h2: int = 50
    ctx_win_size: int = 3
    num_aspects: int = 5
