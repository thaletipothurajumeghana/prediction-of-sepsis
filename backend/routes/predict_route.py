"""
/api/predict  — PDF upload + prediction endpoint
/api/predict/manual — manual values prediction
"""

import os
import uuid
import json
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from utils.pdf_extractor import extract_from_pdf, assess_completeness, get_abnormal_flags, ALL_FEATURES
from utils.ml_engine import predict

predict_bp = Blueprint('predict', __name__)

ALLOWED_EXT = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@predict_bp.route('/upload', methods=['POST'])
def upload_and_extract():
    """
    Step 1: Upload PDF → extract values → return extracted + missing fields.
    Frontend then shows a form to fill in missing values.
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Only PDF files are accepted'}), 400

    # Save uploaded file
    filename   = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    upload_dir = current_app.config['UPLOAD_FOLDER']
    filepath   = os.path.join(upload_dir, filename)
    file.save(filepath)

    # Extract lab values from PDF
    result = extract_from_pdf(filepath)

    if result.get('error'):
        os.remove(filepath)
        # If PDF can't be parsed, return empty extraction (user fills manually)
        return jsonify({
            'success':      True,
            'session_id':   None,
            'extracted':    {},
            'completeness': assess_completeness({}),
            'warning':      f"Could not parse PDF text: {result['error']}. Please enter values manually.",
        })

    extracted    = result['extracted']
    completeness = assess_completeness(extracted)
    abnormals    = get_abnormal_flags(extracted)

    # Store session data
    session_id   = uuid.uuid4().hex
    session_file = os.path.join(upload_dir, f"session_{session_id}.json")
    with open(session_file, 'w') as f:
        json.dump({'extracted': extracted, 'pdf_path': filepath}, f)

    # Clean up PDF after extraction
    try: os.remove(filepath)
    except: pass

    return jsonify({
        'success':      True,
        'session_id':   session_id,
        'extracted':    extracted,
        'completeness': completeness,
        'abnormals':    abnormals,
        'all_features': {k: {'label': v['label'], 'unit': v['unit'],
                              'normal': list(v['normal']), 'group': v['group']}
                         for k, v in ALL_FEATURES.items()},
    })


@predict_bp.route('/predict', methods=['POST'])
def run_prediction():
    """
    Step 2: Receive final values (extracted + user-filled) → run ML prediction.
    """
    body = request.get_json(silent=True) or {}

    values     = body.get('values', {})
    session_id = body.get('session_id')

    # Merge with session-extracted values if session exists
    if session_id:
        upload_dir   = current_app.config['UPLOAD_FOLDER']
        session_file = os.path.join(upload_dir, f"session_{session_id}.json")
        if os.path.exists(session_file):
            with open(session_file) as f:
                session_data = json.load(f)
            merged = {**session_data.get('extracted', {}), **values}
            try: os.remove(session_file)
            except: pass
        else:
            merged = values
    else:
        merged = values

    if not merged:
        return jsonify({'success': False, 'error': 'No patient values provided'}), 400

    # Convert to float where possible
    clean = {}
    for k, v in merged.items():
        try:
            clean[k] = float(v)
        except (TypeError, ValueError):
            pass

    # Check completeness
    completeness = assess_completeness(clean)
    if not completeness['has_enough']:
        return jsonify({
            'success':      False,
            'error':        'not_enough_data',
            'message':      f"Not enough clinical data to make a reliable prediction. "
                            f"Please provide at least {5} of the critical markers: "
                            f"{', '.join(completeness['critical_missing'])}.",
            'completeness': completeness,
        }), 422

    # Run ML prediction
    result = predict(clean)
    if result.get('error'):
        return jsonify({'success': False, 'error': result['error']}), 500

    abnormals = get_abnormal_flags(clean)

    return jsonify({
        'success':         True,
        'prediction':      result,
        'values_used':     clean,
        'completeness':    completeness,
        'abnormal_flags':  abnormals,
        'recommendations': _get_recommendations(result['risk_level'], abnormals),
    })


@predict_bp.route('/predict/manual', methods=['POST'])
def predict_manual():
    """Direct prediction from manually entered values (no PDF)."""
    body   = request.get_json(silent=True) or {}
    values = body.get('values', {})

    clean = {}
    for k, v in values.items():
        try:
            clean[k] = float(v)
        except (TypeError, ValueError):
            pass

    completeness = assess_completeness(clean)
    if not completeness['has_enough']:
        return jsonify({
            'success':  False,
            'error':    'not_enough_data',
            'message':  f"Need at least 5 critical markers. Missing: {', '.join(completeness['critical_missing'])}",
            'completeness': completeness,
        }), 422

    result    = predict(clean)
    abnormals = get_abnormal_flags(clean)

    return jsonify({
        'success':        True,
        'prediction':     result,
        'values_used':    clean,
        'completeness':   completeness,
        'abnormal_flags': abnormals,
        'recommendations': _get_recommendations(result['risk_level'], abnormals),
    })


def _get_recommendations(risk_level: str, abnormals: list) -> list:
    base = {
        'LOW':      ["Continue routine monitoring", "Reassess if clinical condition changes",
                     "Ensure adequate hydration and nutrition"],
        'MODERATE': ["Increase monitoring frequency to every 2–4 hours",
                     "Review antibiotic coverage if infection suspected",
                     "Consider blood cultures if not already obtained",
                     "Monitor urine output closely"],
        'HIGH':     ["Urgent clinical review recommended",
                     "Obtain blood cultures × 2 immediately",
                     "Start broad-spectrum antibiotics within 1 hour per Sepsis-3 bundle",
                     "IV fluid resuscitation: 30 mL/kg crystalloid",
                     "Check lactate level if not recent"],
        'VERY HIGH':["IMMEDIATE MEDICAL ATTENTION REQUIRED",
                     "Activate sepsis protocol / rapid response team",
                     "ICU consult and possible transfer",
                     "Vasopressors if MAP < 65 mmHg despite fluid resuscitation",
                     "Continuous hemodynamic monitoring"],
        'CRITICAL': ["CRITICAL — CALL CODE / EMERGENCY RESPONSE",
                     "Immediate ICU admission",
                     "Vasopressor support likely needed",
                     "Consider intubation if respiratory compromise",
                     "Emergent labs: repeat lactate, ABG, CBC, CMP, coags"],
    }
    recs = base.get(risk_level, base['HIGH'])[:]
    # Add targeted recs for specific abnormals
    ab_feats = {a['feature'] for a in abnormals}
    if 'Lactate' in ab_feats:
        recs.append("Elevated lactate — reassess tissue perfusion and repeat in 2 hours")
    if 'Creatinine' in ab_feats:
        recs.append("Renal impairment detected — monitor urine output, adjust drug doses")
    if 'Platelets' in ab_feats:
        recs.append("Thrombocytopenia — evaluate for DIC, review heparin use")
    if 'Bilirubin_total' in ab_feats:
        recs.append("Hepatic involvement — monitor liver function, avoid hepatotoxic drugs")
    return recs