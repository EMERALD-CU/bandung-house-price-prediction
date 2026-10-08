"""Feature engineering untuk dataset harga rumah Bandung."""

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, load_config

app = typer.Typer()


# =============================================================================
# Feature Engineering
# =============================================================================

def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur turunan yang relevan secara domain:

    - ``price_per_m2``        : Harga per meter persegi luas bangunan
                                (TIDAK disertakan ke model - target leakage)
    - ``building_land_ratio`` : Rasio luas bangunan terhadap luas tanah
    - ``total_rooms``         : Total kamar tidur + kamar mandi
    - ``area_interaction``    : Interaksi luas tanah x luas bangunan
    """
    df = df.copy()

    # Hindari division by zero dengan mengganti 0 menjadi NaN sementara
    building_area_safe = df["building_area"].replace(0, np.nan)
    land_area_safe = df["land_area"].replace(0, np.nan)

    df["price_per_m2"] = df["price"] / building_area_safe
    df["building_land_ratio"] = df["building_area"] / land_area_safe
    df["total_rooms"] = df["bedroom_count"] + df["bathroom_count"]
    df["area_interaction"] = df["land_area"] * df["building_area"]

    # Isi NaN yang muncul dari division by zero dengan median
    for col in ["price_per_m2", "building_land_ratio", "area_interaction"]:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            median_val = df[col].median()
            logger.warning(f"  Fitur '{col}': {n_nan} NaN diisi dengan median ({median_val:.2f})")
            df[col] = df[col].fillna(median_val)

    logger.info(
        "Fitur turunan dibuat: price_per_m2, building_land_ratio, "
        "total_rooms, area_interaction"
    )
    return df


def select_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Pilih kolom final untuk pemodelan berdasarkan konfigurasi.

    Catatan: ``price_per_m2`` DROP dari fitur training karena mengandung
    target leakage (dihitung langsung dari ``price``). Fitur ini tetap
    dihitung untuk keperluan analisis/EDA, tetapi tidak disertakan sebagai
    input model.
    """
    target = config["data"]["target_column"]
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]

    # Hanya sertakan fitur turunan yang TIDAK mengandung target leakage
    safe_engineered_features = [
        "building_land_ratio",
        "total_rooms",
        "area_interaction",
    ]

    keep_cols = [target] + numeric_features + safe_engineered_features + categorical_features
    keep_cols = list(dict.fromkeys(keep_cols))

    df = df[[c for c in keep_cols if c in df.columns]].copy()

    logger.info(f"Kolom final untuk pemodelan: {list(df.columns)}")
    logger.info(f"Shape: {df.shape}")
    logger.info("Catatan: price_per_m2 DROP dari fitur (target leakage).")
    return df


# =============================================================================
# Main Pipeline
# =============================================================================

@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "cleaned.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
) -> None:
    """Pipeline utama: load -> feature engineering -> select -> save."""
    config = load_config()

    logger.info("=== Memulai pipeline feature engineering ===")

    if not input_path.exists():
        raise FileNotFoundError(
            f"File input tidak ditemukan: {input_path}. "
            f"Jalankan 'python -m src.dataset' terlebih dahulu."
        )

    df = pd.read_csv(input_path)
    logger.info(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    df = create_engineered_features(df)
    df = select_features(df, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Fitur disimpan di: {output_path}")


if __name__ == "__main__":
    app()