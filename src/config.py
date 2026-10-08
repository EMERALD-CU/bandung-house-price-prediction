"""Konfigurasi proyek: path constants dan loader YAML dengan override environment variable."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CONFIGS_DIR = PROJ_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "config.yaml"

# If tqdm is installed, configure loguru with tqdm.write
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Memuat konfigurasi YAML dengan override dari environment variable.

    Environment variable dengan prefix ``BHP_`` akan menggantikan nilai
    konfigurasi. Nested key dipisahkan dengan double underscore (``__``).
    Contoh: ``BHP_MODEL__CV_FOLDS=10`` mengubah ``model.cv_folds`` menjadi 10.

    Parameters
    ----------
    config_path : str | Path | None
        Path ke file konfigurasi YAML. Jika ``None``, menggunakan
        ``configs/config.yaml`` relatif terhadap root proyek.

    Returns
    -------
    dict[str, Any]
        Konfigurasi ter-merge.

    Raises
    ------
    FileNotFoundError
        Jika file konfigurasi tidak ditemukan.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"File konfigurasi tidak ditemukan: {config_file}. "
            f"Pastikan file configs/config.yaml sudah dibuat."
        )

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Override dari environment variable
    for key, value in os.environ.items():
        if key.startswith("BHP_"):
            keys = key[4:].lower().split("__")
            target = config
            for k in keys[:-1]:
                target = target.setdefault(k, {})
            target[keys[-1]] = _parse_value(value)

    logger.debug(f"Konfigurasi dimuat dari {config_file}")
    return config


def _parse_value(value: str) -> Any:
    """Konversi string environment variable ke tipe Python yang sesuai."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value