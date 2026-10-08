"""Modul untuk memuat, membersihkan, dan memvalidasi dataset harga rumah Bandung."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR, load_config

app = typer.Typer()


# =============================================================================
# Parser Functions
# =============================================================================

def parse_price(value: str) -> float:
    """
    Parse string harga berformat Indonesia menjadi float (Rupiah).

    Contoh:
        "Rp 2,1 Miliar"  -> 2_100_000_000.0
        "Rp 580 Juta"    -> 580_000_000.0
        "Rp 1,5 Triliun" -> 1_500_000_000_000.0
    """
    if pd.isna(value) or not isinstance(value, str):
        return np.nan

    text = value.lower().strip()

    multiplier = 1.0
    if "triliun" in text:
        multiplier = 1e12
    elif "miliar" in text:
        multiplier = 1e9
    elif "juta" in text:
        multiplier = 1e6
    elif "ribu" in text:
        multiplier = 1e3

    match = re.search(r"([\d.,]+)", text)
    if not match:
        return np.nan

    number_str = match.group(1)

    if "," in number_str and "." in number_str:
        number_str = number_str.replace(".", "").replace(",", ".")
    else:
        number_str = number_str.replace(",", ".")

    try:
        return float(number_str) * multiplier
    except ValueError:
        return np.nan


def parse_area(value: str) -> float:
    """
    Parse string luas berformat '137 m²' menjadi float.

    Contoh:
        "137 m²"   -> 137.0
        "1.500 m²" -> 1500.0
    """
    if pd.isna(value) or not isinstance(value, str):
        return np.nan

    match = re.search(r"([\d.,]+)", value)
    if not match:
        return np.nan

    number_str = match.group(1).replace(".", "").replace(",", ".")

    try:
        return float(number_str)
    except ValueError:
        return np.nan


def parse_location(value: str) -> str:
    """Parse 'Andir, Bandung' menjadi 'Andir'."""
    if pd.isna(value) or not isinstance(value, str):
        return "unknown"
    return value.split(",")[0].strip()


# =============================================================================
# Data Loading
# =============================================================================

def load_raw_dataset(input_path: Path, config: dict) -> pd.DataFrame:
    """Memuat dataset mentah dari CSV tanpa header."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {input_path}.\n"
            f"Unduh dari: https://www.kaggle.com/datasets/khaleeel347/"
            f"harga-rumah-seluruh-kecamatan-di-kota-bandung"
        )

    has_header = config["data"].get("raw_has_header", True)
    raw_columns = config.get("raw_columns")

    if has_header:
        df = pd.read_csv(input_path)
    else:
        df = pd.read_csv(input_path, header=None, names=raw_columns)

    logger.info(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    logger.info(f"Kolom: {list(df.columns)}")
    return df


def validate_schema(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Memvalidasi kolom wajib setelah parsing."""
    required_numeric = config["features"]["numeric"]
    required_categorical = config["features"]["categorical"]
    target = config["data"]["target_column"]

    required_cols = set(required_numeric + required_categorical + [target])
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing_cols}")

    logger.info(f"Validasi skema berhasil: {len(required_cols)} kolom wajib tersedia.")
    return df


# =============================================================================
# Data Cleaning
# =============================================================================

def parse_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Konversi kolom mentah menjadi tipe data yang sesuai."""
    df = df.copy()

    logger.info("Parsing kolom mentah...")
    df["price"] = df["price_raw"].apply(parse_price)
    df["land_area"] = df["land_area_raw"].apply(parse_area)
    df["building_area"] = df["building_area_raw"].apply(parse_area)
    df["location"] = df["location_raw"].apply(parse_location)

    df["carport_count"] = (
        pd.to_numeric(df["carport_count"], errors="coerce").fillna(0).astype(int)
    )

    for col in ["price", "land_area", "building_area"]:
        n_nan = df[col].isna().sum()
        logger.info(f"  {col}: {n_nan} nilai gagal di-parse ({n_nan / len(df):.2%})")

    return df


def clean_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop kolom mentah, filter outlier harga, dan tangani nilai invalid."""
    initial_rows = len(df)

    # Drop kolom mentah yang tidak digunakan
    drop_cols = config["features"].get("drop", [])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Hapus baris dengan target kosong
    target = config["data"]["target_column"]
    df = df.dropna(subset=[target])

    # Kolom yang WAJIB > 0 (harga, luas tanah, luas bangunan)
    strict_positive = config["features"].get("strict_positive", [])
    for col in strict_positive:
        if col in df.columns:
            invalid = (df[col] <= 0).sum()
            if invalid > 0:
                logger.warning(f"  Kolom '{col}': {invalid} nilai <= 0 dihapus")
                df = df[df[col] > 0]

    # Kolom yang boleh bernilai 0 (kamar tidur, kamar mandi, carport)
    allow_zero = config["features"].get("allow_zero", [])
    for col in allow_zero:
        if col in df.columns:
            invalid = (df[col] < 0).sum()
            if invalid > 0:
                logger.warning(f"  Kolom '{col}': {invalid} nilai < 0 dihapus")
                df = df[df[col] >= 0]

    # Filter outlier harga menggunakan quantile
    pre_cfg = config["preprocessing"]
    lower_q = pre_cfg.get("price_lower_quantile", 0.0)
    upper_q = pre_cfg.get("price_upper_quantile", 1.0)

    if lower_q > 0.0 or upper_q < 1.0:
        price_lower = df[target].quantile(lower_q)
        price_upper = df[target].quantile(upper_q)
        n_before = len(df)
        df = df[(df[target] >= price_lower) & (df[target] <= price_upper)]
        n_removed = n_before - len(df)
        logger.info(
            f"  Filter harga [{price_lower:,.0f} – {price_upper:,.0f}] "
            f"({lower_q:.0%}–{upper_q:.0%}): {n_removed} baris dihapus"
        )

    # Hapus duplikat
    df = df.drop_duplicates()

    final_rows = len(df)
    logger.info(
        f"Dataset dibersihkan: {initial_rows} -> {final_rows} baris "
        f"({initial_rows - final_rows} dihapus, "
        f"{(initial_rows - final_rows) / initial_rows:.2%})"
    )
    return df.reset_index(drop=True)


# =============================================================================
# Main Pipeline
# =============================================================================

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "bandung_house_price.csv",
    output_path: Path = INTERIM_DATA_DIR / "cleaned.csv",
) -> None:
    """Pipeline utama: load -> parse -> clean -> save."""
    config = load_config()

    logger.info("=== Memulai pipeline dataset ===")
    df = load_raw_dataset(input_path, config)
    df = parse_columns(df, config)
    df = validate_schema(df, config)
    df = clean_dataset(df, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset bersih disimpan di: {output_path}")
    logger.info(f"Kolom final: {list(df.columns)}")
    logger.info(f"Shape final: {df.shape}")


if __name__ == "__main__":
    app()