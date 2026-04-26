"""
Sepsis Prediction — Inference Script
======================================
Load a trained model and predict sepsis risk for new patient data.

Usage (run from inside backend/ folder):
    python predict.py --input data/sepsis_dataset.csv
    python predict.py --manual
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Resolve models_store relative to THIS file, not the working directory
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models_store')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Clinical reference ranges
REFERENCE_RANGES = {
    'HR':              (60,   100,   'bpm'),
    'O2Sat':           (95,   100,   '%'),
    'Temp':            (36.1, 37.2,  '°C'),
    'SBP':             (90,   140,   'mmHg'),
    'MAP':             (70,   100,   'mmHg'),
    'Resp':            (12,   20,    'breaths/min'),
    'WBC':             (4.5,  11.0,  'x10³/µL'),
    'Lactate':         (0,    2.0,   'mmol/L'),
    'Creatinine':      (0.6,  1.2,   'mg/dL'),
    'Platelets':       (150,  400,   'x10³/µL'),
    'Hgb':             (12,   17.5,  'g/dL'),
    'pH':              (7.35, 7.45,  ''),
    'Glucose':         (70,   110,   'mg/dL'),
    'Bilirubin_total': (0,    1.2,   'mg/dL'),
    'BUN':             (7,    20,    'mg/dL'),
}

RISK_LEVELS = [
    (0.0,  0.2,  'LOW',       'Sepsis unlikely based on current markers.'),
    (0.2,  0.4,  'MODERATE',  'Some indicators present. Monitor closely.'),
    (0.4,  0.6,  'HIGH',      'Significant sepsis risk. Clinical review recommended.'),
    (0.6,  0.8,  'VERY HIGH', 'Strong sepsis indicators. Prompt evaluation needed.'),
    (0.8,  1.01, 'CRITICAL',  'Critical sepsis risk. Immediate medical attention required.'),
]


# ════════════════════════════════════════
# LOAD ARTIFACTS
# ════════════════════════════════════════
def load_artifacts():
    print(f"\n[→] Looking for models in: {MODELS_DIR}")

    if not os.path.isdir(MODELS_DIR):
        print(f"\n[✗] models_store/ folder not found at: {MODELS_DIR}")
        print("    Run  python train.py  first to generate the model files.")
        sys.exit(1)

    required = ['best_model.pkl', 'scaler.pkl', 'imputer.pkl', 'metadata.json']
    for f in required:
        path = os.path.join(MODELS_DIR, f)
        if not os.path.exists(path):
            print(f"\n[✗] Missing file: {path}")
            print("    Run  python train.py  to regenerate all model artifacts.")
            sys.exit(1)

    model   = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    scaler  = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    imputer = joblib.load(os.path.join(MODELS_DIR, 'imputer.pkl'))

    with open(os.path.join(MODELS_DIR, 'metadata.json')) as f:
        meta = json.load(f)

    return model, scaler, imputer, meta


# ════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════
def get_risk_label(prob):
    for lo, hi, label, desc in RISK_LEVELS:
        if lo <= prob < hi:
            return label, desc
    return 'UNKNOWN', ''


def flag_abnormals(patient_dict):
    flags = []
    for feat, (low, high, unit) in REFERENCE_RANGES.items():
        val = patient_dict.get(feat)
        if val is None:
            continue
        try:
            val = float(val)
            if np.isnan(val):
                continue
            if val < low:
                flags.append(f"  ↓ {feat}: {val} {unit}  (low — normal {low}–{high})")
            elif val > high:
                flags.append(f"  ↑ {feat}: {val} {unit}  (high — normal {low}–{high})")
        except (TypeError, ValueError):
            continue
    return flags


def predict_single(patient_dict, model, scaler, imputer, feature_names):
    row = pd.DataFrame([patient_dict])
    for feat in feature_names:
        if feat not in row.columns:
            row[feat] = np.nan
    row = row[feature_names]

    X_imp    = imputer.transform(row)
    X_scaled = scaler.transform(X_imp)
    proba    = float(model.predict_proba(X_scaled)[0, 1])
    pred     = int(proba >= 0.5)
    return pred, proba


def print_result(patient_id, pred, proba, flags):
    risk_label, risk_desc = get_risk_label(proba)
    bar_len = int(proba * 30)
    bar     = '█' * bar_len + '░' * (30 - bar_len)

    COLORS = {
        'LOW':       '\033[92m',   # green
        'MODERATE':  '\033[93m',   # yellow
        'HIGH':      '\033[91m',   # red
        'VERY HIGH': '\033[91m',
        'CRITICAL':  '\033[91m',
    }
    RESET = '\033[0m'
    color = COLORS.get(risk_label, '')

    print(f"\n{'─'*57}")
    print(f"  Patient  : {patient_id}")
    print(f"{'─'*57}")
    print(f"  Result   : {color}{'⚠  SEPSIS DETECTED' if pred else '✓  No Sepsis Detected'}{RESET}")
    print(f"  Risk     : {color}{risk_label}{RESET}")
    print(f"  Prob     : {color}{proba*100:.1f}%{RESET}")
    print(f"  [{bar}] {proba*100:.0f}%")
    print(f"\n  Assessment: {risk_desc}")

    if flags:
        print(f"\n  Abnormal markers ({len(flags)}):")
        for fl in flags:
            print(f"    {fl}")
    else:
        print("\n  All checked markers within normal range.")

    print(f"{'─'*57}")


# ════════════════════════════════════════
# MODES
# ════════════════════════════════════════
def predict_from_csv(path, model, scaler, imputer, meta):
    feature_names = meta['feature_names']

    if not os.path.exists(path):
        print(f"[✗] File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"\n[✓] Loaded {len(df)} patient records from '{path}'")

    results = []
    for i, row in df.iterrows():
        patient_id   = row.get('patient_id', row.get('ID', f"Patient_{i+1}"))
        patient_dict = row.to_dict()
        pred, proba  = predict_single(patient_dict, model, scaler, imputer, feature_names)
        flags        = flag_abnormals(patient_dict)
        risk_label, _= get_risk_label(proba)

        results.append({
            'patient_id':        patient_id,
            'sepsis_prediction': pred,
            'risk_probability':  round(proba, 4),
            'risk_level':        risk_label,
        })
        print_result(patient_id, pred, proba, flags)

    out_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\n[✓] Predictions saved → {out_path}")


def interactive_predict(model, scaler, imputer, meta):
    feature_names = meta['feature_names']

    print("\n" + "="*57)
    print("  MANUAL PATIENT DATA ENTRY")
    print("  Press Enter to skip any field (treated as unknown)")
    print("="*57)

    PROMPTS = [
        ('HR',          'Heart Rate',              'bpm',        60,   100),
        ('O2Sat',       'Oxygen Saturation',        '%',          95,   100),
        ('Temp',        'Temperature',              '°C',         36.1, 37.2),
        ('SBP',         'Systolic BP',              'mmHg',       90,   140),
        ('MAP',         'Mean Arterial Pressure',   'mmHg',       70,   100),
        ('DBP',         'Diastolic BP',             'mmHg',       60,   90),
        ('Resp',        'Respiratory Rate',         '/min',       12,   20),
        ('WBC',         'WBC Count',                'x10³/µL',   4.5,  11.0),
        ('Lactate',     'Lactate',                  'mmol/L',     0,    2.0),
        ('Creatinine',  'Creatinine',               'mg/dL',      0.6,  1.2),
        ('Platelets',   'Platelet Count',           'x10³/µL',   150,  400),
        ('Hgb',         'Hemoglobin',               'g/dL',       12,   17.5),
        ('pH',          'Blood pH',                 '',           7.35, 7.45),
        ('Glucose',     'Glucose',                  'mg/dL',      70,   110),
        ('BUN',         'Blood Urea Nitrogen',      'mg/dL',      7,    20),
        ('Bilirubin_total', 'Total Bilirubin',      'mg/dL',      0,    1.2),
        ('Age',         'Patient Age',              'years',      18,   100),
        ('Gender',      'Gender (0=Female, 1=Male)','',           0,    1),
    ]

    patient_dict = {}
    entered = 0

    for feat, label, unit, lo, hi in PROMPTS:
        hint = f"[normal {lo}–{hi} {unit}]".strip()
        try:
            raw = input(f"  {label:<30} {hint:<22} : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[!] Input cancelled.")
            sys.exit(0)

        if raw:
            try:
                patient_dict[feat] = float(raw)
                entered += 1
            except ValueError:
                print(f"      ⚠  Invalid value '{raw}' — skipping.")
        else:
            patient_dict[feat] = np.nan

    print(f"\n[→] {entered} values entered.")

    if entered < 3:
        print("[✗] Too few values entered. Please provide at least 3 lab values.")
        sys.exit(1)

    pred, proba = predict_single(patient_dict, model, scaler, imputer, feature_names)
    flags       = flag_abnormals(patient_dict)
    print_result("Manual Entry", pred, proba, flags)


# ════════════════════════════════════════
# MAIN
# ════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='SepsisAI — Inference Script',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py --manual\n"
            "  python predict.py --input data/sepsis_dataset.csv\n"
        )
    )
    parser.add_argument('--input',  type=str, help='CSV file with patient rows')
    parser.add_argument('--manual', action='store_true', help='Interactive single-patient entry')
    args = parser.parse_args()

    # Load model
    model, scaler, imputer, meta = load_artifacts()
    best = meta['best_model_name']
    auc  = meta['metrics'][best]['roc_auc']
    print(f"[✓] Model loaded  : {best}")
    print(f"[✓] ROC-AUC       : {auc}")

    if args.manual:
        interactive_predict(model, scaler, imputer, meta)
    elif args.input:
        predict_from_csv(args.input, model, scaler, imputer, meta)
    else:
        parser.print_help()
        print("\n[!] Please pass --manual or --input <file.csv>")


if __name__ == '__main__':
    main()