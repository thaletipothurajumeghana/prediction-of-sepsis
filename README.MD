# SepsisAI — Clinical Prediction Platform

MNC-grade full-stack sepsis prediction application with PDF blood report parsing,
interactive missing value completion, and ML-powered risk assessment.

---

## Tech Stack

| Layer     | Technology                          |
|-----------|-------------------------------------|
| Frontend  | Vanilla HTML/CSS/JS (dark clinical UI) |
| Backend   | Flask 3.0 + Flask-CORS              |
| ML Engine | Scikit-learn + XGBoost + SMOTE      |
| PDF Parse | PyMuPDF (fitz)                      |
| Deploy    | Gunicorn + any Linux server         |

---

## Project Structure

```
sepsis_app/
├── backend/
│   ├── app.py                  ← Flask application factory
│   ├── train.py                ← ML training pipeline
│   ├── models_store/           ← Saved model artifacts
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   ├── imputer.pkl
│   │   └── metadata.json
│   ├── routes/
│   │   ├── predict_route.py    ← /api/upload, /api/predict
│   │   ├── health_route.py     ← /api/health
│   │   └── report_route.py
│   ├── utils/
│   │   ├── pdf_extractor.py    ← PDF parsing + lab value extraction
│   │   └── ml_engine.py        ← Model inference engine
│   └── uploads/                ← Temporary file storage
│
├── frontend/
│   ├── templates/
│   │   └── index.html          ← Single-page application
│   └── static/
│       ├── css/main.css        ← Dark clinical UI styles
│       └── js/app.js           ← SPA logic
│
├── requirements.txt
├── start.sh
└── README.md
```

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train model (if not already trained)
cd backend
python train.py       # needs data/sepsis_dataset.csv
cd ..

# 3. Run
bash start.sh         # starts Flask on port 5000
# OR
cd backend && python app.py
```

Open: **http://localhost:5000**

---

## Application Flow

```
User → Upload PDF → Flask /api/upload
                  → PyMuPDF extracts text
                  → Regex parses 36 lab values
                  → Returns: extracted + missing fields
                  ↓
User → Fills missing values in form
     → Flask /api/predict
     → ML Engine: impute → scale → predict
     → Returns: probability, risk level, components
                  ↓
User sees: Risk verdict, breakdown, findings, recommendations
```

---

## API Endpoints

| Method | Endpoint             | Description                          |
|--------|----------------------|--------------------------------------|
| GET    | `/api/health`        | Server + model status                |
| POST   | `/api/upload`        | Upload PDF, returns extracted values |
| POST   | `/api/predict`       | Run prediction with merged values    |
| POST   | `/api/predict/manual`| Predict from manual values only      |

### POST /api/upload
```
Content-Type: multipart/form-data
Body: file=<PDF>

Response:
{
  "success": true,
  "session_id": "abc123",
  "extracted": { "HR": 92, "WBC": 14.2, ... },
  "completeness": {
    "found_count": 18,
    "completeness_pct": 50,
    "has_enough": true,
    "critical_missing": ["Lactate", "MAP"]
  },
  "all_features": { ... feature definitions ... }
}
```

### POST /api/predict
```json
{
  "session_id": "abc123",
  "values": { "Lactate": 3.2, "MAP": 58 }
}

Response:
{
  "success": true,
  "prediction": {
    "probability": 72.4,
    "risk_level": "HIGH",
    "sepsis_detected": true,
    "confidence": 72.4,
    "risk_components": [...]
  },
  "abnormal_flags": [...],
  "recommendations": [...]
}
```

---

## Missing Data Handling

The app handles incomplete blood reports intelligently:

1. **PDF upload** → extracts whatever values are readable
2. **Completeness check** → identifies critical missing markers (WBC, Lactate, Creatinine, etc.)
3. **Interactive form** → shows all 36 fields with:
   - Green border = extracted from PDF
   - Red border = critical missing (must fill)
   - Gray border = optional
4. **Validation** → requires ≥ 5 critical markers before prediction
5. **"Not enough data" response** → explains exactly what's missing

---

## Production Deployment

```bash
# Gunicorn (production WSGI)
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 "app:create_app()"

# With nginx reverse proxy
upstream sepsisai {
    server 127.0.0.1:5000;
}
server {
    listen 80;
    server_name yourdomain.com;
    location / { proxy_pass http://sepsisai; }
    client_max_body_size 20M;
}
```

---

## Dataset

Download from Kaggle:
```bash
kaggle datasets download -d salikhussaini49/prediction-of-sepsis --unzip -p backend/data/
mv backend/data/*.csv backend/data/sepsis_dataset.csv
cd backend && python train.py
```
URL: https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis

---

## References

- PhysioNet/Computing in Cardiology Challenge 2019
- Sepsis-3 Clinical Criteria — Singer et al., JAMA 2016
- SOFA Score — Vincent et al., ICM 1996
