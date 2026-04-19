"""
frontend/ui_components.py
All reusable UI helpers, CSS injection, API calls, and render functions.
Imported by streamlit_app.py – no Streamlit state is held here.
"""
from __future__ import annotations
import io
import requests
import streamlit as st

# ── Palette & CSS ─────────────────────────────────────────────────────────────

LOGO_HTML = """
<div style="text-align:center; padding: 8px 0 4px;">
  <span style="font-family:'Georgia',serif; font-size:1.35rem;
               letter-spacing:.05em; color:#E2F0FF; font-weight:700;">
    Q-Neuro<span style="color:#4FC3F7;">Detect</span>
  </span><br>
  <span style="font-size:.72rem; color:#78909C; letter-spacing:.12em;
               text-transform:uppercase;">Parkinson · ZSK Solutions</span>
</div>
"""


def page_header() -> str:
    return """
<div style="padding: 1.2rem 0 .4rem;">
  <h1 style="font-family:'Georgia',serif; font-size:2rem; margin:0;
             color:#E2F0FF; letter-spacing:.02em;">
    🧠 Q-NeuroDetect <span style="color:#4FC3F7;">Parkinson</span>
  </h1>
  <p style="color:#78909C; font-size:.88rem; margin:.3rem 0 0;">
    Çok modlu yapay zeka destekli Parkinson karar destek sistemi &nbsp;·&nbsp;
    Ses &nbsp;·&nbsp; MRI &nbsp;·&nbsp; Çizim &nbsp;·&nbsp; Füzyon
  </p>
</div>
"""


def modality_card_header(icon: str, name: str, formats: str) -> str:
    return f"""
<div style="background:#0D1B2A; border:1px solid #1E3A5F; border-radius:8px;
            padding:.65rem 1rem .5rem; margin-bottom:.4rem;">
  <span style="font-size:1.1rem;">{icon}</span>
  <span style="font-family:'Georgia',serif; font-size:1rem;
               color:#E2F0FF; margin-left:.4rem; font-weight:600;">{name}</span><br>
  <span style="font-size:.72rem; color:#546E7A; letter-spacing:.06em;">{formats}</span>
</div>
"""


def inject_css() -> None:
    st.markdown("""
<style>
/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'Georgia', serif;
}
.stApp {
  background: #07111C;
  color: #CFD8DC;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: #0B1929 !important;
  border-right: 1px solid #1E3A5F;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0277BD 0%, #01579B 100%);
  color: #E2F0FF;
  border: none;
  border-radius: 6px;
  font-family: 'Georgia', serif;
  font-size: .95rem;
  letter-spacing: .04em;
  padding: .65rem 2rem;
  transition: filter .2s;
}
.stButton > button[kind="primary"]:hover {
  filter: brightness(1.18);
}

/* ── Uploader ── */
[data-testid="stFileUploader"] {
  background: #0D1B2A;
  border: 1px dashed #1E3A5F;
  border-radius: 8px;
}

/* ── Metric ── */
[data-testid="stMetric"] {
  background: #0D1B2A;
  border: 1px solid #1E3A5F;
  border-radius: 8px;
  padding: .5rem .8rem;
}

/* ── Tabs ── */
.stTabs [role="tab"] { font-family: 'Georgia', serif; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #4FC3F7 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #07111C; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── API Calls ──────────────────────────────────────────────────────────────────

def fetch_health(base_url: str, timeout: int) -> dict | None:
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def call_predict_all(
    base_url: str,
    timeout: int,
    voice_file,
    mri_file,
    drawing_file,
    fusion_method: str,
    generate_report: bool,
) -> tuple[dict | None, str | None]:
    files = {}
    if voice_file:
        files["voice_file"] = (voice_file.name, voice_file.getvalue(), voice_file.type or "audio/wav")
    if mri_file:
        files["mri_file"] = (mri_file.name, mri_file.getvalue(), mri_file.type or "image/png")
    if drawing_file:
        files["drawing_file"] = (drawing_file.name, drawing_file.getvalue(), drawing_file.type or "image/png")

    params = {"fusion_method": fusion_method, "generate_report": str(generate_report).lower()}
    try:
        r = requests.post(
            f"{base_url}/predict/all",
            files=files or None,
            params=params,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json(), None
    except requests.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)


def fetch_report_bytes(base_url: str, report_url: str) -> bytes | None:
    try:
        full_url = base_url + report_url
        r = requests.get(full_url, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception:
        return None


# ── Render Helpers ─────────────────────────────────────────────────────────────

def render_health_badge(health: dict | None) -> None:
    if health and health.get("status") == "ok":
        loaded = health.get("models_loaded", {})
        n_loaded = sum(1 for v in loaded.values() if v)
        n_total = len(loaded)
        st.success(f"✅ Bağlı · v{health.get('version','?')} · {n_loaded}/{n_total} model")
        if loaded:
            for name, ok in loaded.items():
                icon = "🟢" if ok else "⚪"
                st.caption(f"{icon} {name}")
    else:
        st.error("❌ Backend'e ulaşılamıyor")


def render_risk_banner(fusion: dict) -> None:
    score = fusion["score"]
    level = fusion["level"]
    label = fusion["label"]
    pct = int(score * 100)

    color_map = {
        "High": ("#B71C1C", "#FFCDD2", "🔴"),
        "Moderate": ("#E65100", "#FFE0B2", "🟡"),
        "Low": ("#1B5E20", "#C8E6C9", "🟢"),
    }
    bg, fg, icon = color_map.get(level, ("#263238", "#ECEFF1", "⚪"))

    bar_color = {"High": "#EF5350", "Moderate": "#FFA726", "Low": "#66BB6A"}.get(level, "#78909C")

    st.markdown(f"""
<div style="background:{bg}22; border:1px solid {bg}88;
            border-radius:10px; padding:1.2rem 1.6rem; margin-bottom:.5rem;">
  <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
    <span style="font-size:2.2rem;">{icon}</span>
    <div>
      <div style="font-family:'Georgia',serif; font-size:1.6rem;
                  color:{fg}; font-weight:700; line-height:1.1;">
        {level} Risk &nbsp;·&nbsp; {label}
      </div>
      <div style="color:#90A4AE; font-size:.85rem; margin-top:.2rem;">
        Füzyon risk skoru
      </div>
    </div>
    <div style="margin-left:auto; text-align:right;">
      <div style="font-family:'Georgia',serif; font-size:2.4rem;
                  color:{fg}; font-weight:700;">{pct}%</div>
    </div>
  </div>
  <!-- progress bar -->
  <div style="background:#1E3A5F; border-radius:4px; height:8px; margin-top:.9rem;">
    <div style="background:{bar_color}; border-radius:4px;
                height:8px; width:{pct}%; transition:width .6s ease;"></div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_modality_results(modalities: list[dict]) -> None:
    _ICONS = {"voice": "🎙️", "mri": "🧲", "drawing": "✏️"}
    _MODEL_BADGE = {
        "deep_learning": ("DL", "#0277BD"),
        "classical_ml": ("ML", "#558B2F"),
        "quantum": ("Q", "#6A1B9A"),
    }

    for m in modalities:
        pct = int(m["probability"] * 100)
        conf = int(m["confidence"] * 100)
        is_pk = m["label"] == "Parkinson"
        bar_c = "#EF5350" if is_pk else "#66BB6A"
        label_c = "#EF9A9A" if is_pk else "#A5D6A7"
        icon = _ICONS.get(m["modality"], "🔬")
        badge_txt, badge_c = _MODEL_BADGE.get(m["model_type"], ("?", "#546E7A"))
        features_str = ", ".join(m.get("features_used", [])[:4])
        notes = m.get("notes") or ""

        st.markdown(f"""
<div style="background:#0D1B2A; border:1px solid #1E3A5F; border-radius:8px;
            padding:.85rem 1.1rem; margin-bottom:.6rem;">
  <div style="display:flex; align-items:center; gap:.6rem; margin-bottom:.5rem;">
    <span style="font-size:1.2rem;">{icon}</span>
    <span style="font-family:'Georgia',serif; font-size:1rem;
                 color:#E2F0FF; font-weight:600; text-transform:capitalize;">
      {m['modality']}
    </span>
    <span style="background:{badge_c}33; color:{badge_c}; border:1px solid {badge_c}66;
                 border-radius:4px; font-size:.68rem; padding:.1rem .4rem;
                 letter-spacing:.06em; margin-left:.2rem;">{badge_txt}</span>
    <span style="margin-left:auto; font-family:'Georgia',serif;
                 font-size:1.15rem; font-weight:700; color:{label_c};">
      {m['label']}
    </span>
  </div>
  <div style="background:#07111C; border-radius:4px; height:6px; margin-bottom:.45rem;">
    <div style="background:{bar_c}; border-radius:4px;
                height:6px; width:{pct}%;"></div>
  </div>
  <div style="display:flex; gap:1.5rem; font-size:.8rem; color:#78909C;">
    <span>Olasılık <b style="color:#B0BEC5;">{pct}%</b></span>
    <span>Güven <b style="color:#B0BEC5;">{conf}%</b></span>
    {'<span>' + notes + '</span>' if notes else ''}
  </div>
  {f'<div style="font-size:.72rem; color:#546E7A; margin-top:.3rem;">Özellikler: {features_str}</div>' if features_str else ''}
</div>
""", unsafe_allow_html=True)


def render_model_comparison(comparison: list[dict]) -> None:
    _TYPE_COLOR = {
        "deep_learning": "#4FC3F7",
        "classical_ml": "#AED581",
        "quantum": "#CE93D8",
    }
    _TYPE_LABEL = {
        "deep_learning": "Derin Öğrenme",
        "classical_ml": "Klasik ML",
        "quantum": "Kuantum",
    }

    for c in comparison:
        pct = int(c["probability"] * 100)
        tc = _TYPE_COLOR.get(c["model_type"], "#78909C")
        tl = _TYPE_LABEL.get(c["model_type"], c["model_type"])
        is_pk = c["label"] == "Parkinson"
        bar_c = "#EF5350" if is_pk else "#66BB6A"

        st.markdown(f"""
<div style="background:#0D1B2A; border:1px solid #1E3A5F; border-radius:7px;
            padding:.65rem .9rem; margin-bottom:.5rem;">
  <div style="display:flex; align-items:center; gap:.5rem; margin-bottom:.35rem;">
    <span style="font-size:.78rem; color:{tc}; letter-spacing:.05em;
                 font-weight:600;">{tl}</span>
    <span style="font-size:.78rem; color:#546E7A; flex:1;">{c['model_name']}</span>
    <span style="font-size:.9rem; font-weight:700;
                 color:{'#EF9A9A' if is_pk else '#A5D6A7'};">{pct}%</span>
  </div>
  <div style="background:#07111C; border-radius:3px; height:4px;">
    <div style="background:{bar_c}; border-radius:3px; height:4px; width:{pct}%;"></div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_explanation(text: str) -> None:
    lines = text.strip().split("\n")
    parts = []
    for line in lines:
        if line.startswith("•"):
            parts.append(f'<li style="margin:.25rem 0; color:#90A4AE;">{line[1:].strip()}</li>')
        elif line.strip():
            parts.append(f'<p style="margin:.4rem 0; color:#B0BEC5;">{line}</p>')

    inner = "\n".join(parts)
    # Wrap bullet items
    inner = inner.replace(
        '<li', '<ul style="padding-left:1.2rem; margin:.3rem 0;"><li', 1
    )
    if "</li>" in inner:
        inner = inner.rsplit("</li>", 1)
        inner = "</li>".join(inner) + "</ul>" if len(inner) > 1 else inner[0]

    st.markdown(f"""
<div style="background:#0B1929; border-left:3px solid #0277BD;
            border-radius:0 8px 8px 0; padding:1rem 1.2rem; line-height:1.65;">
  {text.replace(chr(10), '<br>')}
</div>
""", unsafe_allow_html=True)


def render_report_download(base_url: str, report_url: str) -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    pdf_bytes = fetch_report_bytes(base_url, report_url)
    if pdf_bytes:
        st.download_button(
            label="📄 PDF Raporu İndir",
            data=pdf_bytes,
            file_name="q_neurodetect_rapor.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning("PDF rapor henüz hazır değil veya backend'e ulaşılamıyor.")


# ── Phase 3: SHAP + Model Comparison Chart ────────────────────────────────────

def render_shap_bar(shap_values: dict[str, float]) -> None:
    """SHAP özellik önem çubuğu — st.bar_chart ile basit görsel."""
    import streamlit as st
    import pandas as pd

    if not shap_values:
        st.caption("SHAP verisi mevcut değil (model yüklü değil).")
        return

    df = pd.DataFrame(
        {"Özellik": list(shap_values.keys()),
         "SHAP Değeri": list(shap_values.values())}
    ).sort_values("SHAP Değeri", key=abs, ascending=True)

    st.markdown("""
<div style="background:#0D1B2A; border:1px solid #1E3A5F; border-radius:8px;
            padding:.8rem 1rem .5rem; margin-bottom:.6rem;">
  <span style="color:#4FC3F7; font-size:.85rem; letter-spacing:.06em;">
    ⚡ SHAP — En Etkili 10 Özellik (Ses Modeli)
  </span>
</div>
""", unsafe_allow_html=True)

    colors = ["#EF5350" if v > 0 else "#66BB6A" for v in df["SHAP Değeri"]]
    st.bar_chart(df.set_index("Özellik")["SHAP Değeri"])
    st.caption("🔴 Pozitif = Parkinson'a işaret eder  ·  🟢 Negatif = Healthy'e işaret eder")


def render_training_guide() -> None:
    """Sidebar veya expander içinde eğitim adımlarını göster."""
    import streamlit as st

    with st.expander("📚 Model Eğitimi — Başlangıç Kılavuzu", expanded=False):
        st.markdown("""
**Veri hazırlığı:**

```
data/raw/voice/
    labels.csv        ← filename, label (0/1)
    audio/
        hasta001.wav

data/raw/mri/
    healthy/  *.png
    parkinson/ *.png
```

**Eğitim komutları** (proje kökünden):
```bash
# 1. Ses modeli (SVM + RF ensemble)
python notebooks/train_voice.py

# 2. MRI modeli (MobileNetV2)
python notebooks/train_mri.py

# 3. QSVM (PennyLane)
python notebooks/train_qsvm.py
```

**Eğitimden sonra backend'i yeniden başlat:**
```bash
# Ctrl+C → tekrar:
uvicorn app.main:app --reload --port 8000
```

Modeller `models/voice/` ve `models/mri/` klasörlerine kaydedilir ve
backend otomatik olarak yükler.
""")


# ── Phase 5: Contribution Cards + Fusion Method Badge ─────────────────────────

def render_contribution_cards(contributions: dict[str, float],
                               modalities: list[dict]) -> None:
    import streamlit as st

    if not contributions:
        return

    st.markdown("""
<div style="background:#0D1B2A; border:1px solid #1E3A5F; border-radius:8px;
            padding:.75rem 1rem .5rem; margin-bottom:.5rem;">
  <span style="color:#4FC3F7; font-size:.85rem; letter-spacing:.06em;">
    📊 Modalite Katkı Yüzdeleri
  </span>
</div>
""", unsafe_allow_html=True)

    _ICONS = {"voice": "🎙️", "mri": "🧲", "drawing": "✏️"}
    prob_map = {m["modality"]: m["probability"] for m in modalities}

    cols = st.columns(len(contributions))
    for col, (mod, pct) in zip(cols, contributions.items()):
        prob = prob_map.get(mod, 0.5)
        bar_c = "#EF5350" if prob >= 0.5 else "#66BB6A"
        icon  = _ICONS.get(mod, "🔬")
        with col:
            st.markdown(f"""
<div style="background:#07111C; border:1px solid #1E3A5F; border-radius:8px;
            padding:.8rem .6rem; text-align:center;">
  <div style="font-size:1.6rem;">{icon}</div>
  <div style="font-family:'Georgia',serif; font-size:1.3rem;
              font-weight:700; color:#E2F0FF; margin:.2rem 0;">
    {pct:.0f}%
  </div>
  <div style="font-size:.75rem; color:#78909C; text-transform:capitalize;
              margin-bottom:.5rem;">{mod}</div>
  <div style="background:#1E3A5F; border-radius:3px; height:5px;">
    <div style="background:{bar_c}; border-radius:3px;
                height:5px; width:{pct}%;"></div>
  </div>
  <div style="font-size:.72rem; color:#546E7A; margin-top:.3rem;">
    P={prob:.0%}
  </div>
</div>
""", unsafe_allow_html=True)


def render_risk_factors(risk_factors: dict) -> None:
    import streamlit as st

    if not risk_factors:
        return

    risks  = risk_factors.get("risk", [])
    prot   = risk_factors.get("protective", [])

    col_r, col_p = st.columns(2)
    with col_r:
        if risks:
            st.markdown("**🔴 Risk Bulguları**")
            for r in risks:
                st.markdown(f"""
<div style="background:#B71C1C18; border-left:3px solid #EF5350;
            border-radius:0 6px 6px 0; padding:.4rem .7rem;
            margin:.3rem 0; font-size:.83rem; color:#FFCDD2;">
  {r}
</div>""", unsafe_allow_html=True)

    with col_p:
        if prot:
            st.markdown("**🟢 Koruyucu Bulgular**")
            for p in prot:
                st.markdown(f"""
<div style="background:#1B5E2018; border-left:3px solid #66BB6A;
            border-radius:0 6px 6px 0; padding:.4rem .7rem;
            margin:.3rem 0; font-size:.83rem; color:#C8E6C9;">
  {p}
</div>""", unsafe_allow_html=True)


def render_fusion_badge(method: str, missing: list[str], calibrated: bool) -> None:
    import streamlit as st

    method_labels = {
        "weighted": "Weighted Ensemble",
        "stacking": "Stacking Meta-Learner",
        "bayesian": "Bayesian Averaging",
        "weighted_ensemble": "Weighted Ensemble",
        "bayesian_avg": "Bayesian Averaging",
    }
    label = method_labels.get(method, method)
    cal_badge = "✓ Kalibre" if calibrated else "ham skor"

    miss_html = ""
    if missing:
        miss_html = (f'<span style="color:#FFA726; font-size:.75rem; margin-left:.8rem;">'
                     f'⚠️ Eksik: {", ".join(missing)}</span>')

    st.markdown(f"""
<div style="background:#0D1B2A; border:1px solid #1E3A5F; border-radius:6px;
            padding:.45rem .9rem; margin-bottom:.5rem; font-size:.8rem;">
  <span style="color:#4FC3F7;">⚙️ Füzyon:</span>
  <span style="color:#B0BEC5; margin-left:.4rem;">{label}</span>
  <span style="color:#66BB6A; margin-left:.8rem; font-size:.72rem;">
    {cal_badge}
  </span>
  {miss_html}
</div>
""", unsafe_allow_html=True)
