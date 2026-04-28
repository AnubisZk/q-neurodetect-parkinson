"""
Q-NeuroDetect Parkinson – Streamlit Frontend  (Phase 2)

Çalıştırma (proje kökünden):
    streamlit run frontend/streamlit_app.py

Backend must be running at BACKEND_URL (default https://q-neurodetect-parkinson-production.up.railway.app).
"""

import os
import sys
from pathlib import Path

# Proje kökünü sys.path'e ekle — nerede çalıştırılırsa çalıştırılsın import çalışsın
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import streamlit as st

# ui_components: proje kökünden veya frontend/ içinden çalışma senaryolarını destekle
try:
    import frontend.ui_components as ui   # proje kökünden: streamlit run frontend/streamlit_app.py
except ModuleNotFoundError:
    import ui_components as ui             # frontend/ içinden: streamlit run streamlit_app.py

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = "https://q-neurodetect-parkinson-production.up.railway.app"
TIMEOUT = 30

st.set_page_config(
    page_title="Q-NeuroDetect Parkinson",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_css()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(ui.LOGO_HTML, unsafe_allow_html=True)
    st.markdown("---")
    fusion_method = st.radio(
        "Füzyon Yöntemi",
        ["weighted", "stacking", "bayesian"],
        format_func=lambda x: {"meta": "Meta Classifier (Stacking)", "weighted": "Weighted Ensemble", "bayesian": "Bayesian Averaging"}.get(x, x),
        help="meta: eğitilmiş füzyon modeli · weighted: ağırlıklı ortalama",
    )
    generate_report = st.toggle("PDF Rapor Oluştur", value=True)
    st.markdown("---")
    st.caption("**Backend**")
    health = ui.fetch_health(BACKEND_URL, TIMEOUT)
    ui.render_health_badge(health)
    st.markdown("---")
    ui.render_training_guide()
    st.caption("© Q-NeuroDetect — Parkinson / ZSK Solutions")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(ui.page_header(), unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────────────────────────
st.markdown("### 📂 Veri Yükleme")
col_v, col_m, col_d = st.columns(3)

with col_v:
    st.markdown(ui.modality_card_header("🎙️", "Ses", "WAV · MP3 · FLAC · OGG"), unsafe_allow_html=True)
    voice_file = st.file_uploader(
        "Ses dosyası", type=["wav", "mp3", "flac", "ogg"],
        key="voice", label_visibility="collapsed",
    )
    if voice_file:
        st.audio(voice_file)

with col_m:
    st.markdown(ui.modality_card_header("🧲", "MRI", "NIfTI · PNG · JPG · DCM"), unsafe_allow_html=True)
    mri_file = st.file_uploader(
        "MRI dosyası", type=["nii", "png", "jpg", "jpeg", "dcm"],
        key="mri", label_visibility="collapsed",
    )
    if mri_file and mri_file.type.startswith("image"):
        st.image(mri_file, width=300)

with col_d:
    st.markdown(ui.modality_card_header("✏️", "Çizim", "PNG · JPG · CSV"), unsafe_allow_html=True)
    drawing_file = st.file_uploader(
        "Çizim dosyası", type=["png", "jpg", "jpeg", "bmp", "csv"],
        key="drawing", label_visibility="collapsed",
    )
    if drawing_file and drawing_file.type.startswith("image"):
        st.image(drawing_file, width=300)

st.markdown("<br>", unsafe_allow_html=True)

# ── Analyse Button ────────────────────────────────────────────────────────────
any_uploaded = any([voice_file, mri_file, drawing_file])
run_label = "🔬 Analizi Çalıştır" if any_uploaded else "🔬 Dosya yüklenmeden çalıştır (mock)"

if st.button(run_label, type="primary", width=300):
    with st.spinner("Analiz yapılıyor…"):
        result, error = ui.call_predict_all(
            BACKEND_URL, TIMEOUT,
            voice_file, mri_file, drawing_file,
            fusion_method, generate_report,
        )

    if error:
        st.error(f"❌ Backend hatası: {error}")
        st.stop()

    st.session_state["result"] = result

# ── Results ───────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    st.markdown("---")
    st.markdown("### 📊 Analiz Sonuçları")

    ui.render_risk_banner(result["fusion"])
    ui.render_fusion_badge(
        result.get("fusion_method", "weighted"),
        result.get("missing_modalities", []),
        calibrated=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Katkı kartları
    if result.get("modality_contributions"):
        ui.render_contribution_cards(
            result["modality_contributions"], result["modalities"])
        st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        st.markdown("#### Modalite Bazlı Tahminler")
        ui.render_modality_results(result["modalities"])

    with right:
        st.markdown("#### Model Karşılaştırması")
        ui.render_model_comparison(result["model_comparison"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📝 Açıklama")
    ui.render_explanation(result["explanation"])

    if result.get("risk_factors"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ⚖️ Risk & Koruyucu Bulgular")
        ui.render_risk_factors(result["risk_factors"])
    st.caption(f"Request ID: `{result['request_id']}`")

    if result.get("report_url"):
        ui.render_report_download(BACKEND_URL, result["report_url"])

    # SHAP — opsiyonel, model yüklü ise göster
    if result.get("shap_values"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ⚡ SHAP Açıklanabilirlik")
        ui.render_shap_bar(result["shap_values"])
