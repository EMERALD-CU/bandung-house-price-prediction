"""Training pipeline untuk model prediksi harga rumah Bandung."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, load_config

app = typer.Typer()


# =============================================================================
# Model Registry
# =============================================================================

MODEL_REGISTRY = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


# =============================================================================
# Preprocessor Builder
# =============================================================================

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    missing_strategy: str = "median",
) -> ColumnTransformer:
    """Bangun ColumnTransformer untuk numerik (impute + scale) dan kategorikal (impute + onehot)."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=missing_strategy)),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


# =============================================================================
# Training Pipeline
# =============================================================================

def train_pipeline(config: dict, features_path: Path, model_dir: Path) -> dict:
    """
    Eksekusi pipeline training lengkap dengan log-transform pada target.
    """
    # 1. Load dataset
    if not features_path.exists():
        raise FileNotFoundError(
            f"File fitur tidak ditemukan: {features_path}. "
            f"Jalankan 'python -m src.features' terlebih dahulu."
        )

    df = pd.read_csv(features_path)
    logger.info(f"Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    target = config["data"]["target_column"]
    numeric_features = config["features"]["numeric"]
    engineered_features = ["building_land_ratio", "total_rooms", "area_interaction"]
    categorical_features = config["features"]["categorical"]

    all_numeric = list(dict.fromkeys(numeric_features + engineered_features))

    missing_cols = set([target] + all_numeric + categorical_features) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Kolom tidak ditemukan di features.csv: {missing_cols}")

    X = df[all_numeric + categorical_features]
    y = df[target]

    # 2. Split data SEBELUM preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["preprocessing"]["test_size"],
        random_state=config["project"]["random_state"],
    )
    logger.info(f"Split: train={len(X_train)}, test={len(X_test)}")

    # 3. Log-transform pada target jika diaktifkan
    use_log = config["preprocessing"].get("log_transform_target", False)
    if use_log:
        y_train_fit = np.log1p(y_train)
        logger.info("Log-transform diterapkan pada target (np.log1p).")
    else:
        y_train_fit = y_train

    # 4. Preprocessing pipeline
    preprocessor = build_preprocessor(
        numeric_features=all_numeric,
        categorical_features=categorical_features,
        missing_strategy=config["preprocessing"]["missing_strategy"],
    )

    # 5. Hyperparameter tuning
    best_score = np.inf
    best_model_name = None
    best_estimator = None
    best_params = None
    results_per_model = {}

    for candidate in config["model"]["candidates"]:
        name = candidate["name"]
        param_grid_raw = candidate["params"]
        model_class = MODEL_REGISTRY[name]

        logger.info(f"Tuning model: {name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model_class(random_state=config["project"]["random_state"])),
        ])

        param_grid = {f"model__{k}": v for k, v in param_grid_raw.items()}

        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=config["model"]["cv_folds"],
            scoring=config["model"]["scoring"],
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_train, y_train_fit)
        cv_score = -grid_search.best_score_

        results_per_model[name] = {
            "cv_mape": float(cv_score),
            "best_params": grid_search.best_params_,
        }

        logger.info(f"  {name}: MAPE CV (log-space) = {cv_score:.4f}")

        if cv_score < best_score:
            best_score = cv_score
            best_model_name = name
            best_estimator = grid_search.best_estimator_
            best_params = grid_search.best_params_

    logger.info(f"Model terbaik: {best_model_name} (MAPE CV = {best_score:.4f})")

    # 6. Evaluasi pada test set (konversi kembali ke skala asli)
    y_pred_fit = best_estimator.predict(X_test)
    if use_log:
        y_pred = np.expm1(y_pred_fit)
    else:
        y_pred = y_pred_fit

    metrics = {
        "model_name": best_model_name,
        "best_params": {k.replace("model__", ""): v for k, v in best_params.items()},
        "log_transform": use_log,
        "cv_mape_log_space": float(best_score),
        "test_mape": float(mean_absolute_percentage_error(y_test, y_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_rmse": float(root_mean_squared_error(y_test, y_pred)),
        "test_r2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "all_models": results_per_model,
    }

    logger.info("=== Metrik Evaluasi Test Set ===")
    logger.info(f"  R2   : {metrics['test_r2']:.4f}")
    logger.info(f"  MAPE : {metrics['test_mape']:.2%}")
    logger.info(f"  MAE  : Rp {metrics['test_mae']:,.0f}")
    logger.info(f"  RMSE : Rp {metrics['test_rmse']:,.0f}")

    # 7. Simpan artifact
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / config["artifacts"]["model_filename"]
    metrics_path = model_dir / config["artifacts"]["metrics_filename"]

    joblib.dump(best_estimator, model_path)
    logger.success(f"Model disimpan di: {model_path}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.success(f"Metrics disimpan di: {metrics_path}")

    return metrics


# =============================================================================
# Main CLI
# =============================================================================

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_dir: Path = MODELS_DIR,
) -> None:
    """Pipeline utama: load features -> training -> save artifacts."""
    config = load_config()

    logger.info("=== Memulai training pipeline ===")
    train_pipeline(config, features_path, model_dir)


if __name__ == "__main__":
    app()