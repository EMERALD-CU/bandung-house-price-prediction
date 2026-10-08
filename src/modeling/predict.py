"""Inference pipeline untuk model prediksi harga rumah Bandung."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, load_config

app = typer.Typer()


def load_model_and_metrics(model_dir: Path, config: dict):
    """Load model artifact dan metrics dari disk."""
    model_path = model_dir / config["artifacts"]["model_filename"]
    metrics_path = model_dir / config["artifacts"]["metrics_filename"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di {model_path}. "
            f"Jalankan 'python -m src.modeling.train' terlebih dahulu."
        )

    model = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    logger.info(f"Model dimuat: {metrics.get('model_name', 'unknown')}")
    return model, metrics


def predict(model, input_df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Prediksi harga untuk DataFrame input.

    Parameters
    ----------
    model : estimator
        Pipeline sklearn yang sudah di-fit.
    input_df : pd.DataFrame
        DataFrame dengan kolom fitur (numerik + engineered + kategorikal).
    config : dict
        Konfigurasi proyek.

    Returns
    -------
    np.ndarray
        Array harga prediksi (Rupiah, skala asli).
    """
    input_df = input_df.copy()

    # Pastikan fitur turunan ada (jika belum, hitung dari fitur dasar)
    if "building_land_ratio" not in input_df.columns:
        input_df["building_land_ratio"] = (
            input_df["building_area"] / input_df["land_area"].replace(0, np.nan)
        ).fillna(0.95)
    if "total_rooms" not in input_df.columns:
        input_df["total_rooms"] = input_df["bedroom_count"] + input_df["bathroom_count"]
    if "area_interaction" not in input_df.columns:
        input_df["area_interaction"] = input_df["land_area"] * input_df["building_area"]

    y_pred = model.predict(input_df)

    # Konversi dari log-space jika perlu
    if config["preprocessing"].get("log_transform_target", False):
        y_pred = np.expm1(y_pred)

    return y_pred


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_dir: Path = MODELS_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
    n_samples: int = 10,
) -> None:
    """
    Demo inference: load model, ambil n_samples dari dataset, bandingkan
    prediksi dengan aktual, dan simpan hasil prediksi untuk seluruh dataset.
    """
    config = load_config()
    logger.info("=== Memulai pipeline inference ===")

    model, metrics = load_model_and_metrics(model_dir, config)

    if not features_path.exists():
        raise FileNotFoundError(
            f"File fitur tidak ditemukan: {features_path}. "
            f"Jalankan 'python -m src.features' terlebih dahulu."
        )

    df = pd.read_csv(features_path)
    logger.info(f"Dataset dimuat: {df.shape[0]} baris")

    # Ambil sampel untuk demo
    sample = df.sample(
        n=min(n_samples, len(df)),
        random_state=config["project"]["random_state"],
    )
    target_col = config["data"]["target_column"]
    X_sample = sample.drop(columns=[target_col])
    y_true = sample[target_col].values
    y_pred = predict(model, X_sample, config)

    result = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "error_pct": (y_pred - y_true) / y_true * 100,
    })

    logger.info("=== Perbandingan Harga Aktual vs Prediksi (10 sampel) ===")
    for _, row in result.iterrows():
        logger.info(
            f"  Actual: Rp {row['actual']:>15,.0f}  |  "
            f"Predicted: Rp {row['predicted']:>15,.0f}  |  "
            f"Error: {row['error_pct']:>7.2f}%"
        )

    mean_abs_pct_error = result["error_pct"].abs().mean()
    logger.info(
        f"Rata-rata absolute % error pada {len(result)} sampel: "
        f"{mean_abs_pct_error:.2f}%"
    )

    # Simpan seluruh prediksi pada dataset lengkap
    X_full = df.drop(columns=[target_col])
    y_pred_full = predict(model, X_full, config)
    df["predicted_price"] = y_pred_full
    df["error_pct"] = (df["predicted_price"] - df[target_col]) / df[target_col] * 100
    df.to_csv(output_path, index=False)
    logger.success(f"Prediksi disimpan di: {output_path}")


if __name__ == "__main__":
    app()