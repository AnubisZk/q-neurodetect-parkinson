# Q-NeuroDetect Parkinson 🧠

**Çok modlu Parkinson Karar Destek Sistemi**  
Ses · MRI · Çizim → Weighted/Meta Füzyon → Risk Skoru + PDF Rapor

---

## Mimari Özet

```
parkinson_multimodal_system/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, CORS, routes
│   ├── routes/
│   │   ├── predict_voice.py     # POST /predict/voice
│   │   ├── predict_mri.py       # POST /predict/mri
│   │   ├── predict_drawing.py   # POST /predict/drawing
│   │   └── predict_fusion.py    # POST /predict/all
│   ├── services/
│   │   ├── model_loader.py      # .h5 / .pkl lazy loader
│   │   ├── preprocessing_voice.py
│   │   ├── preprocessing_mri.py
│   │   ├── preprocessing_drawing.py
│   │   ├── feature_engineering.py
│   │   ├── fusion_engine.py     # weighted avg + meta-classifier
│   │   ├── explainability.py    # Türkçe açıklama üretici
│   │   └── report_generator.py  # ReportLab PDF
│   ├── schemas/
│   │   └── prediction_schema.py # Pydantic v2 response models
│   └── utils/
│       ├── config.py            # Pydantic Settings (.env destekli)
│       ├── validators.py        # Uzantı / boyut doğrulama
│       └── file_handlers.py     # Async upload → UUID dosya adı
├── frontend/                    # Phase 2: Streamlit/Gradio arayüzü
├── models/
│   ├── voice/   ← voice_model.h5    buraya
│   ├── mri/     ← mri_model.h5      buraya
│   ├── drawing/ ← drawing_model.pkl buraya
│   └── fusion/  ← fusion_model.pkl  buraya
├── notebooks/   # Eğitim notebook'ları (Phase 2)
├── data/
│   ├── raw/ · processed/ · uploads/ · reports/
├── requirements.txt
└── README.md
```

---

## Kurulum

### 1. Ortam oluştur

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **TensorFlow olmadan çalıştırmak için** `requirements.txt` içinde `tensorflow` satırını yoruma alın.  
> Tüm modül mock inference ile çalışır.

### 2. (İsteğe bağlı) .env yapılandırması

```ini
# .env
DEBUG=true
VOICE_MODEL_PATH=models/voice/voice_model.h5
MRI_MODEL_PATH=models/mri/mri_model.h5
DRAWING_MODEL_PATH=models/drawing/drawing_model.pkl
FUSION_MODEL_PATH=models/fusion/fusion_model.pkl
FUSION_WEIGHTS=[0.35,0.40,0.25]
HIGH_RISK_THRESHOLD=0.70
MODERATE_RISK_THRESHOLD=0.40
```

### 3. Sunucuyu başlat

```bash
# Proje kökünden:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Endpoint'leri

| Method | URL | Açıklama |
|--------|-----|----------|
| `GET`  | `/health` | Sistem durumu, model yükleme bilgisi |
| `POST` | `/predict/voice` | Ses dosyasından Parkinson tahmini |
| `POST` | `/predict/mri` | MRI dosyasından tahmin |
| `POST` | `/predict/drawing` | Çizim testinden tahmin |
| `POST` | `/predict/all` | Tüm modaliteler + füzyon + PDF |

### Swagger UI

```
http://localhost:8000/docs
```

### ReDoc

```
http://localhost:8000/redoc
```

---

## Örnek İstekler

### GET /health
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": {
    "voice": false,
    "mri": false,
    "drawing": false,
    "fusion": false
  }
}
```

### POST /predict/voice
```bash
curl -X POST http://localhost:8000/predict/voice \
  -F "file=@sample.wav"
```
```json
{
  "modality": "voice",
  "probability": 0.7231,
  "label": "Parkinson",
  "confidence": 0.4462,
  "model_type": "deep_learning",
  "features_used": ["jitter_pct", "shimmer_db", "hnr", "rpde", "dfa", "spread1", "spread2", "ppe"],
  "notes": "HNR=0.412, Jitter=1.823%"
}
```

### POST /predict/all (füzyon)
```bash
curl -X POST http://localhost:8000/predict/all \
  -F "voice_file=@sample.wav" \
  -F "mri_file=@scan.png" \
  -F "drawing_file=@spiral.png" \
  -F "generate_report=true" \
  -F "fusion_method=meta"
```

---

## Model Entegrasyonu

Gerçek modelleri kullanmak için ilgili klasöre dosyaları koyun:

```
models/voice/voice_model.h5       # tf.keras CNN modeli
models/mri/mri_model.h5           # tf.keras CNN modeli
models/drawing/drawing_model.pkl  # scikit-learn Pipeline/Classifier
models/fusion/fusion_model.pkl    # scikit-learn meta-classifier
```

Dosyalar bulunamazsa sistem otomatik olarak mock inference'a geçer — geliştirme ortamı için API her koşulda çalışır.

---

## Füzyon Yöntemleri

| Yöntem | Parametre | Açıklama |
|--------|-----------|----------|
| Weighted Average | `fusion_method=weighted` | `FUSION_WEIGHTS` [ses, mri, çizim] ile ağırlıklı ortalama |
| Meta Classifier | `fusion_method=meta` | Eğitilmiş sklearn modeli; yoksa weighted'a döner |

---

## Risk Seviyeleri

| Skor | Seviye | Etiket |
|------|--------|--------|
| ≥ 0.70 | 🔴 High | Parkinson |
| 0.40 – 0.69 | 🟡 Moderate | Uncertain |
| < 0.40 | 🟢 Low | Healthy |

---

## Yol Haritası

- **Phase 1** ✅ FastAPI backend, mock inference, PDF rapor iskeleti
- **Phase 2** ✅ Streamlit arayüzü — upload widget'ları, risk banner, model karşılaştırma, PDF indirme
- **Phase 3** 🔜 Gerçek model entegrasyonu, QSVM (PennyLane/Qiskit)
- **Phase 4** 🔜 Docker, CI/CD, hasta geçmişi kaydı

---

## Frontend (Phase 2)

```bash
# Backend çalışırken ayrı bir terminalde:
streamlit run frontend/streamlit_app.py

# Farklı backend adresi için:
BACKEND_URL=http://my-server:8000 streamlit run frontend/streamlit_app.py
```

Arayüz açılır: `http://localhost:8501`

---

## Testler

```bash
pytest tests/ -v
```

```bash
# Hızlı smoke test (server çalışırken):
curl http://localhost:8000/health
```

---

## Lisans

MIT © Q-NeuroDetect — Parkinson / ZSK Solutions
