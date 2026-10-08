"""Aplikasi Streamlit untuk prediksi harga rumah di Bandung."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, load_config
from src.modeling.predict import predict

# =============================================================================
# Konfigurasi Halaman
# =============================================================================

st.set_page_config(
    page_title="Prediksi Harga Rumah Bandung",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS untuk Tampilan Modern
# =============================================================================

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f1f5f9;
        border-radius: 12px;
        padding: 1.25rem;
        border-left: 4px solid #3b82f6;
    }
    .prediction-box {
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 1rem;
        opacity: 0.85;
        margin-bottom: 0.5rem;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Load Artifacts (Cached)
# =============================================================================

@st.cache_resource
def load_artifacts():
    """Load model, metrics, dan dataset referensi dari disk."""
    config = load_config()

    model_path = MODELS_DIR / config["artifacts"]["model_filename"]
    metrics_path = MODELS_DIR / config["artifacts"]["metrics_filename"]
    features_path = PROCESSED_DATA_DIR / "features.csv"

    if not model_path.exists():
        st.error(
            f"Model tidak ditemukan di `{model_path}`. "
            f"Jalankan `python -m src.modeling.train` terlebih dahulu."
        )
        st.stop()

    model = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    df = pd.read_csv(features_path) if features_path.exists() else None

    return model, metrics, df, config


model, metrics, df, config = load_artifacts()

# Daftar kecamatan unik (dari data)
LOCATIONS = sorted(df["location"].dropna().unique().tolist()) if df is not None else ["Andir"]


# =============================================================================
# Format Helper
# =============================================================================

def format_rupiah(value: float) -> str:
    """Format angka ke Rupiah dengan pemisah ribuan."""
    if abs(value) >= 1e12:
        return f"Rp {value / 1e12:.2f} triliun"
    if abs(value) >= 1e9:
        return f"Rp {value / 1e9:.2f} miliar"
    if abs(value) >= 1e6:
        return f"Rp {value / 1e6:.0f} juta"
    return f"Rp {value:,.0f}".replace(",", ".")


# =============================================================================
# Header
# =============================================================================

st.markdown('<div class="main-header">🏠 Prediksi Harga Rumah Bandung</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Model machine learning untuk estimasi harga properti berdasarkan "
    "karakteristik fisik dan lokasi kecamatan di Kota Bandung."
    "</div>",
    unsafe_allow_html=True,
)

# =============================================================================
# Sidebar — Input Form
# =============================================================================

with st.sidebar:
    st.header("📝 Input Properti")

    location = st.selectbox(
        "Kecamatan",
        options=LOCATIONS,
        index=0,
        help="Pilih kecamatan lokasi properti.",
    )

    col1, col2 = st.columns(2)
    with col1:
        bedroom_count = st.number_input("Kamar Tidur", min_value=0, max_value=15, value=3, step=1)
        carport_count = st.number_input("Carport", min_value=0, max_value=10, value=1, step=1)
    with col2:
        bathroom_count = st.number_input("Kamar Mandi", min_value=0, max_value=15, value=2, step=1)

    land_area = st.number_input(
        "Luas Tanah (m²)",
        min_value=10, max_value=10000, value=120, step=10,
    )
    building_area = st.number_input(
        "Luas Bangunan (m²)",
        min_value=10, max_value=10000, value=100, step=10,
    )

    predict_btn = st.button("🔮 Prediksi Harga", type="primary", use_container_width=True)

# =============================================================================
# Main Area — Hasil Prediksi
# =============================================================================

col_result, col_info = st.columns([2, 1])

with col_result:
    if predict_btn:
        # Validasi input
        if building_area > land_area * 3:
            st.warning(
                "⚠️ Luas bangunan jauh lebih besar dari luas tanah. "
                "Prediksi mungkin kurang akurat."
            )

        # Bangun DataFrame input
        input_df = pd.DataFrame([{
            "bedroom_count": bedroom_count,
            "bathroom_count": bathroom_count,
            "carport_count": carport_count,
            "land_area": land_area,
            "building_area": building_area,
            "location": location,
        }])

        # Prediksi
        with st.spinner("Menghitung estimasi harga..."):
            prediction = predict(model, input_df, config)[0]

        # Tampilkan hasil
        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-label">Estimasi Harga Properti</div>
                <div class="prediction-value">{format_rupiah(prediction)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metrik turunan
        st.markdown("### 📊 Detail Estimasi")
        col_a, col_b, col_c = st.columns(3)
        price_per_m2 = prediction / building_area if building_area > 0 else 0
        ratio = building_area / land_area if land_area > 0 else 0

        col_a.metric("Harga per m²", format_rupiah(price_per_m2))
        col_b.metric("Rasio Bangunan/Tanah", f"{ratio:.2f}")
        col_c.metric("Total Ruangan", f"{bedroom_count + bathroom_count}")

        # Perbandingan dengan median kecamatan
        if df is not None:
            median_loc = df[df["location"] == location]["price"].median()
            median_all = df["price"].median()
            st.markdown("### 📍 Konteks Pasar")
            col_d, col_e = st.columns(2)
            col_d.metric(
                f"Median di {location}",
                format_rupiah(median_loc) if not pd.isna(median_loc) else "N/A",
            )
            col_e.metric("Median Bandung", format_rupiah(median_all))

            diff_pct = (prediction - median_loc) / median_loc * 100 if median_loc > 0 else 0
            if diff_pct > 10:
                st.info(f"💰 Properti ini diprediksi **{diff_pct:.1f}%** di atas median kecamatan.")
            elif diff_pct < -10:
                st.info(f"💸 Properti ini diprediksi **{abs(diff_pct):.1f}%** di bawah median kecamatan.")
            else:
                st.success(f"✅ Properti ini diprediksi mendekati median kecamatan (selisih {abs(diff_pct):.1f}%).")
    else:
        st.info("👈 Masukkan detail properti di panel kiri, lalu klik **Prediksi Harga**.")
        if df is not None:
            st.markdown("### 📈 Distribusi Harga di Bandung")
            fig = px.histogram(
                df,
                x="price",
                nbins=60,
                title="Distribusi Harga Rumah (skala log)",
                labels={"price": "Harga (Rp)"},
            )
            fig.update_layout(
                xaxis=dict(type="log"),
                showlegend=False,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.markdown("### 🎯 Performa Model")
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; color: #64748b;">Model</div>
            <div style="font-size: 1.25rem; font-weight: 600; color: #1e293b;">
                {metrics.get("model_name", "N/A").replace("_", " ").title()}
            </div>
        </div>
        <br>
        """,
        unsafe_allow_html=True,
    )

    st.metric("R² Score", f"{metrics.get('test_r2', 0):.4f}")
    st.metric("MAPE", f"{metrics.get('test_mape', 0):.2%}")
    st.metric("MAE", format_rupiah(metrics.get("test_mae", 0)))
    st.metric("RMSE", format_rupiah(metrics.get("test_rmse", 0)))

    st.markdown("### 📚 Tentang Dataset")
    if df is not None:
        st.markdown(
            f"- **Total properti**: {len(df):,}\n"
            f"- **Kecamatan**: {df['location'].nunique()}\n"
            f"- **Sumber**: [Kaggle](https://www.kaggle.com/datasets/khaleeel347/"
            f"harga-rumah-seluruh-kecamatan-di-kota-bandung)"
        )

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.caption(
    "© 2026 Emerald Alphante Reirezqi · "
    "Dibangun dengan Streamlit · Scikit-Learn · Plotly · "
    "[GitHub Repository](https://github.com/emerald-alpha/bandung-house-price-prediction)"
)