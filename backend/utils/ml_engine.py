"""
ML Inference Engine — loads trained model and runs sepsis prediction
"""

import os
import json
import numpy as np
import joblib

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models_store')

_model   = None
_scaler  = None
_imputer = None
_meta    = None


def _load_artifacts():
    global _model, _scaler, _imputer, _meta
    if _model is not None:
        return True
    try:
        _model   = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
        _scaler  = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        _imputer = joblib.load(os.path.join(MODELS_DIR, 'imputer.pkl'))
        with open(os.path.join(MODELS_DIR, 'metadata.json')) as f:
            _meta = json.load(f)
        return True
    except Exception as e:
        print(f"[Model load error] {e}")
        return False


def predict(patient_values: dict) -> dict:
    """
    Run sepsis prediction on a patient values dict.
    Returns full prediction result with risk breakdown.
    """
    if not _load_artifacts():
        return {'error': 'Model not loaded. Run training first.'}

    feature_names = _meta['feature_names']
    import pandas as pd

    row = pd.DataFrame([patient_values])
    for feat in feature_names:
        if feat not in row.columns:
            row[feat] = np.nan
    row = row[feature_names]

    X_imp    = _imputer.transform(row)
    X_scaled = _scaler.transform(X_imp)
    proba    = float(_model.predict_proba(X_scaled)[0, 1])
    pred     = int(proba >= 0.5)

    # Risk breakdown — weighted contribution of feature groups
    risk_components = _compute_risk_components(patient_values, proba)

    risk_level, risk_color = _get_risk_level(proba)

    return {
        'prediction':        pred,
        'probability':       round(proba * 100, 1),
        'risk_level':        risk_level,
        'risk_color':        risk_color,
        'risk_components':   risk_components,
        'model_name':        _meta.get('best_model_name', 'Ensemble'),
        'model_auc':         _meta['metrics'][_meta['best_model_name']]['roc_auc'],
        'sepsis_detected':   bool(pred),
        'confidence':        round(max(proba, 1 - proba) * 100, 1),
    }


def _get_risk_level(prob):
    if prob < 0.20:  return 'LOW',       '#16a34a'
    if prob < 0.40:  return 'MODERATE',  '#ca8a04'
    if prob < 0.60:  return 'HIGH',      '#ea580c'
    if prob < 0.80:  return 'VERY HIGH', '#dc2626'
    return              'CRITICAL',      '#991b1b'


def _compute_risk_components(values: dict, overall_prob: float) -> list:
    """Compute approximate risk contributions by clinical domain."""

    def score(keys, hi_bad=True):
        """Score 0–100 based on how many keys are abnormal."""
        from backend.utils.pdf_extractor import ALL_FEATURES
        pts = []
        for k in keys:
            if k not in values or values[k] is None:
                continue
            try:
                v   = float(values[k])
                lo, hi = ALL_FEATURES[k]['normal']
                rng = hi - lo
                if hi_bad:
                    if v > hi:   pts.append(min((v - hi) / (rng * 0.5 + 0.001), 1.0))
                    elif v < lo: pts.append(min((lo - v) / (rng * 0.5 + 0.001), 1.0))
                    else:        pts.append(0.0)
                else:
                    pts.append(0.0)
            except Exception:
                continue
        return round(np.mean(pts) * 100, 1) if pts else round(overall_prob * 60, 1)

    inflammatory = score(['WBC', 'Temp', 'HR', 'Resp'])
    organ_dys    = score(['Creatinine', 'Bilirubin_total', 'Platelets', 'Lactate'])
    infection    = score(['WBC', 'Lactate', 'pH', 'HCO3'])
    hemodynamic  = score(['SBP', 'MAP', 'HR', 'Lactate'])

    # Blend with overall probability for calibration
    def blend(raw):
        return round(raw * 0.6 + overall_prob * 100 * 0.4, 1)

    return [
        {'label': 'Inflammatory markers',  'value': blend(inflammatory), 'level': _level(blend(inflammatory))},
        {'label': 'Organ dysfunction',     'value': blend(organ_dys),    'level': _level(blend(organ_dys))},
        {'label': 'Infection probability', 'value': blend(infection),    'level': _level(blend(infection))},
        {'label': 'Hemodynamic instability','value': blend(hemodynamic), 'level': _level(blend(hemodynamic))},
    ]


def _level(v):
    if v < 25:  return 'low'
    if v < 50:  return 'medium'
    if v < 75:  return 'high'
    return 'critical'


def model_info() -> dict:
    if not _load_artifacts():
        return {'loaded': False}
    return {
        'loaded':     True,
        'model_name': _meta.get('best_model_name'),
        'metrics':    _meta.get('metrics', {}),
        'features':   len(_meta.get('feature_names', [])),
    }