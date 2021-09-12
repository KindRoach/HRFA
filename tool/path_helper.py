from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


def create_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def create_folder(path):
    path.mkdir(parents=True, exist_ok=True)
