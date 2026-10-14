from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # root folder of the project


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(PROJECT_ROOT / ".env")  # read local environment variables

DATA_DIR = Path(os.environ.get("DFD_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
ARTIFACTS_DIR = Path(os.environ.get("DFD_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")).expanduser()  # model weights
REPORTS_DIR = PROJECT_ROOT / "reports"  # benchmark metrics
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_CONFIG_PATH = Path(os.environ.get("DFD_DATA_CONFIG", CONFIGS_DIR / "data.yaml")).expanduser()
APP_DATA_DIR = Path(os.environ.get("DFD_APP_DATA_DIR", PROJECT_ROOT / "app_data")).expanduser()
HISTORY_DB_PATH = Path(os.environ.get("DFD_HISTORY_DB", APP_DATA_DIR / "history.db")).expanduser()  # sqlite db path

PROCESSED_ROOT = Path(os.environ.get("DFD_PROCESSED_ROOT", DATA_DIR)).expanduser()
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_DIR = DATA_DIR / "manifests"
CACHE_DIR = DATA_DIR / "cache"

CANVAS_SIZE = 256  # standard canvas size
CROP_SIZE = 224  # crop size for backbone input


def load_yaml(path: str | Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)  # parse yaml config
