# 🏠 Bandung House Price Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bandung-house-price-prediction.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/emerald-alpha/bandung-house-price-model)

End-to-end machine learning pipeline untuk memprediksi harga rumah di Kota Bandung berdasarkan karakteristik fisik properti dan lokasi kecamatan.

## 🚀 Live Demo

**Coba aplikasi:** [bandung-house-price-prediction.streamlit.app](https://bandung-house-price-prediction.streamlit.app)

Aplikasi Streamlit interaktif yang memungkinkan pengguna memasukkan karakteristik properti (kecamatan, jumlah kamar, luas tanah/bangunan) dan mendapatkan estimasi harga secara real-time.

## 📊 Performa Model

| Metrik | Nilai |
|---|---|
| **Model** | Gradient Boosting |
| **R² Score** | 0.8423 |
| **MAPE** | 27.77% |
| **MAE** | Rp 1.06 miliar |
| **RMSE** | Rp 2.21 miliar |

**Hyperparameter terbaik:** `learning_rate=0.1`, `max_depth=7`, `n_estimators=200`

**Model selection:** Dibandingkan 3 kandidat algoritma (Ridge, Random Forest, Gradient Boosting) dengan 5-fold cross-validation dan scoring MAPE. Gradient Boosting unggul tipis atas Random Forest (0.0119 vs 0.0121 pada log-space MAPE).

**Model artifact:** Di-host di [Hugging Face Hub](https://huggingface.co/emerald-alpha/bandung-house-price-model) — bukan di repository ini, untuk menjaga repo tetap ringan.

## 📁 Dataset

- **Sumber:** [Kaggle — khaleeel347](https://www.kaggle.com/datasets/khaleeel347/harga-rumah-seluruh-kecamatan-di-kota-bandung)
- **Total data mentah:** 7.611 properti
- **Setelah cleaning:** 6.095 properti (19.92% dihapus)
- **Kecamatan:** 27 di Kota Bandung
- **Fitur final:** 9 (5 dasar + 3 engineered + 1 kategorikal)

**Fitur:**
- `bedroom_count`, `bathroom_count`, `carport_count` — jumlah kamar & carport
- `land_area`, `building_area` — luas tanah & bangunan (m²)
- `building_land_ratio` — rasio luas bangunan terhadap luas tanah
- `total_rooms` — total kamar tidur + kamar mandi
- `area_interaction` — interaksi luas tanah × luas bangunan
- `location` — kecamatan (kategorikal)

**Target:** `price` (Rupiah)

## 🏗️ Arsitektur Pipeline

```
┌──────────────────┐
│  Raw CSV         │  7.611 baris × 11 kolom (tanpa header)
└────────┬─────────┘
         │ src/dataset.py
         │  - Parse price: "Rp 2,1 Miliar" → 2100000000.0
         │  - Parse area: "137 m²" → 137.0
         │  - Parse location: "Andir, Bandung" → "Andir"
         │  - Filter outlier harga (1%–99% quantile)
         ▼
┌──────────────────┐
│  data/interim/   │  6.095 baris × 7 kolom
│  cleaned.csv     │
└────────┬─────────┘
         │ src/features.py
         │  - Feature engineering (3 fitur turunan)
         │  - Drop price_per_m2 (target leakage)
         ▼
┌──────────────────┐
│  data/processed/ │  6.095 baris × 10 kolom
│  features.csv    │
└────────┬─────────┘
         │ src/modeling/train.py
         │  - Train-test split 80:20 (sebelum preprocessing)
         │  - Log-transform target (np.log1p)
         │  - ColumnTransformer: StandardScaler + OneHotEncoder
         │  - GridSearchCV per model (5-fold CV)
         ▼
┌──────────────────┐
│  models/         │  best_model.joblib + metrics.json
└────────┬─────────┘
         │ src/modeling/predict.py + app/streamlit_app.py
         ▼
┌──────────────────┐
│  Streamlit Cloud │  Interactive web app
└──────────────────┘
```

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Data manipulation** | Python 3.10+, Pandas, NumPy |
| **Machine learning** | Scikit-Learn (Ridge, RandomForest, GradientBoosting) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Deployment** | Streamlit Community Cloud, Hugging Face Hub |
| **Configuration** | PyYAML, python-dotenv |
| **Logging** | Loguru |
| **Serialization** | Joblib |
| **Linting/Formatting** | Ruff |
| **Testing** | Pytest, Pytest-cov |
| **Package management** | Setuptools, pip |

## 🏃 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/EMERALD-CU/bandung-house-price-prediction.git
cd bandung-house-price-prediction
```

### 2. Setup Environment

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
pip install -e .
```

### 3. Unduh Dataset

Unduh dari [Kaggle](https://www.kaggle.com/datasets/khaleeel347/harga-rumah-seluruh-kecamatan-di-kota-bandung) dan letakkan file `results.csv` di `data/raw/bandung_house_price.csv`.

Atau gunakan Kaggle API:

```bash
kaggle datasets download -d khaleeel347/harga-rumah-seluruh-kecamatan-di-kota-bandung -p data/raw --unzip
```

### 4. Jalankan Pipeline

```bash
# Step 1: Load & clean dataset
python -m src.dataset

# Step 2: Feature engineering
python -m src.features

# Step 3: Training (GridSearchCV, ~1-3 menit)
python -m src.modeling.train

# Step 4: Demo inference
python -m src.modeling.predict
```

### 5. Jalankan Aplikasi Streamlit

```bash
streamlit run app/streamlit_app.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

## 📂 Struktur Proyek

```
bandung-house-price-prediction/
├── app/
│   └── streamlit_app.py         # Aplikasi UI deployment
├── configs/
│   └── config.yaml              # Konfigurasi hyperparameter & path
├── data/
│   ├── raw/                     # Dataset mentah (gitignored)
│   ├── interim/                 # Setelah cleaning (gitignored)
│   └── processed/               # Fitur siap model
├── models/                      # Model artifacts (gitignored, di HF Hub)
├── scripts/
│   └── show_metrics.py          # Utilitas inspect metrics
├── src/
│   ├── config.py                # Loader konfigurasi YAML + path constants
│   ├── dataset.py               # Load, parse, clean dataset
│   ├── features.py              # Feature engineering
│   ├── plots.py                 # Fungsi visualisasi
│   └── modeling/
│       ├── train.py             # Training pipeline + GridSearchCV
│       └── predict.py           # Inference
├── tests/                       # Unit test (pytest)
├── .gitignore
├── LICENSE                      # MIT
├── Makefile                     # Convenience commands
├── pyproject.toml               # Package config + ruff + pytest
├── requirements.txt
└── README.md
```

## 🔬 Detail Teknis

### Penanganan Target Leakage

Fitur `price_per_m2` = `price / building_area` dihitung untuk keperluan EDA, tetapi **tidak disertakan** dalam training karena mengandung target leakage. Jika disertakan, model akan "menghafal" hubungan trivial dan menghasilkan metrik yang menyesatkan.

### Log-Transform Target

Distribusi harga properti sangat **right-skewed** (median Rp 2,3 miliar, max Rp 40 miliar setelah outlier removal). Transformasi `np.log1p(price)` menormalkan distribusi dan:
- Menurunkan MAPE dari 161% → 27.77%
- Menaikkan R² dari -2.24 → 0.8423
- Menurunkan RMSE dari Rp 21,7 miliar → Rp 2,21 miliar

### Outlier Removal

Harga mentah memiliki outlier ekstrem (max Rp 735 miliar untuk properti 56 m² — jelas error). Filter quantile 1%–99% membuang 146 baris bermasalah.

### Preprocessing Pipeline

`StandardScaler` dan `OneHotEncoder` di-fit **hanya pada training set** (setelah `train_test_split`) untuk mencegah data leakage. Seluruh pipeline dibungkus dalam objek `Pipeline` sehingga saat inference cukup `model.predict(X_raw)`.

## 📄 Lisensi

MIT License — lihat [LICENSE](LICENSE)

## 👤 Author

**Emerald Alphante Reirezqi**
- Email: emeraldalphanter@gmail.com
- LinkedIn: [linkedin.com/in/emerald-ar](https://linkedin.com/in/emerald-ar)
- GitHub: [github.com/EMERALD-CU](https://github.com/EMERALD-CU)