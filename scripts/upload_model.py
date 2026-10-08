"""Upload model artifacts ke Hugging Face Hub."""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from loguru import logger

load_dotenv()

HF_REPO_ID = os.getenv("HF_REPO_ID", "emerald-alpha/bandung-house-price-model")
MODELS_DIR = Path("models")


def main() -> None:
    """Upload semua file di folder models/ ke Hugging Face Hub."""
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {MODELS_DIR}")

    files = [f for f in MODELS_DIR.glob("*") if f.is_file() and f.name != ".gitkeep"]
    if not files:
        raise FileNotFoundError(f"Tidak ada file di {MODELS_DIR} untuk di-upload.")

    logger.info(f"Repo ID: {HF_REPO_ID}")
    logger.info(f"File yang akan di-upload: {[f.name for f in files]}")

    api = HfApi()

    create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
    logger.info(f"Repo siap: https://huggingface.co/{HF_REPO_ID}")

    for file in files:
        logger.info(f"Meng-upload {file.name}...")
        api.upload_file(
            path_or_fileobj=str(file),
            path_in_repo=file.name,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
        logger.success(f"  {file.name} berhasil di-upload")

    logger.success(f"Semua model tersedia di: https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()