"""
PDF Extractor — parses blood report PDFs and extracts lab values
"""

import re
import os
import json
import numpy as np

# ── All features the model knows about
ALL_FEATURES = {
    # Vital Signs
    'HR':           {'label': 'Heart Rate',              'unit': 'bpm',       'normal': (60, 100),    'group': 'vitals'},
    'O2Sat':        {'label': 'Oxygen Saturation',       'unit': '%',         'normal': (95, 100),    'group': 'vitals'},
    'Temp':         {'label': 'Temperature',             'unit': '°C',        'normal': (36.1, 37.2), 'group': 'vitals'},
    'SBP':          {'label': 'Systolic BP',             'unit': 'mmHg',      'normal': (90, 140),    'group': 'vitals'},
    'MAP':          {'label': 'Mean Arterial Pressure',  'unit': 'mmHg',      'normal': (70, 100),    'group': 'vitals'},
    'DBP':          {'label': 'Diastolic BP',            'unit': 'mmHg',      'normal': (60, 90),     'group': 'vitals'},
    'Resp':         {'label': 'Respiratory Rate',        'unit': '/min',      'normal': (12, 20),     'group': 'vitals'},

    # Critical Labs
    'WBC':          {'label': 'White Blood Cells',       'unit': 'x10³/µL',  'normal': (4.5, 11.0),  'group': 'critical'},
    'Lactate':      {'label': 'Lactate',                 'unit': 'mmol/L',    'normal': (0.5, 2.0),   'group': 'critical'},
    'Creatinine':   {'label': 'Creatinine',              'unit': 'mg/dL',     'normal': (0.6, 1.2),   'group': 'critical'},
    'Platelets':    {'label': 'Platelet Count',          'unit': 'x10³/µL',  'normal': (150, 400),   'group': 'critical'},
    'Bilirubin_total': {'label': 'Total Bilirubin',      'unit': 'mg/dL',     'normal': (0.2, 1.2),   'group': 'critical'},

    # Blood Chemistry
    'Hgb':          {'label': 'Hemoglobin',              'unit': 'g/dL',      'normal': (12.0, 17.5), 'group': 'cbc'},
    'Hct':          {'label': 'Hematocrit',              'unit': '%',         'normal': (37, 52),     'group': 'cbc'},
    'pH':           {'label': 'Blood pH',                'unit': '',          'normal': (7.35, 7.45), 'group': 'abg'},
    'PaCO2':        {'label': 'PaCO2',                   'unit': 'mmHg',      'normal': (35, 45),     'group': 'abg'},
    'HCO3':         {'label': 'Bicarbonate (HCO3)',      'unit': 'mmol/L',    'normal': (22, 26),     'group': 'abg'},
    'BaseExcess':   {'label': 'Base Excess',             'unit': 'mmol/L',    'normal': (-2, 2),      'group': 'abg'},
    'FiO2':         {'label': 'FiO2',                    'unit': 'fraction',  'normal': (0.21, 0.40), 'group': 'abg'},

    # Metabolic
    'Glucose':      {'label': 'Glucose',                 'unit': 'mg/dL',     'normal': (70, 110),    'group': 'metabolic'},
    'BUN':          {'label': 'Blood Urea Nitrogen',     'unit': 'mg/dL',     'normal': (7, 20),      'group': 'metabolic'},
    'Sodium':       {'label': 'Sodium',                  'unit': 'mEq/L',     'normal': (136, 145),   'group': 'metabolic'},
    'Potassium':    {'label': 'Potassium',               'unit': 'mEq/L',     'normal': (3.5, 5.0),   'group': 'metabolic'},
    'Chloride':     {'label': 'Chloride',                'unit': 'mEq/L',     'normal': (98, 107),    'group': 'metabolic'},
    'Calcium':      {'label': 'Calcium',                 'unit': 'mg/dL',     'normal': (8.5, 10.5),  'group': 'metabolic'},
    'Magnesium':    {'label': 'Magnesium',               'unit': 'mEq/L',     'normal': (1.7, 2.2),   'group': 'metabolic'},
    'Phosphate':    {'label': 'Phosphate',               'unit': 'mg/dL',     'normal': (2.5, 4.5),   'group': 'metabolic'},

    # Liver / Coag
    'AST':          {'label': 'AST (Liver)',             'unit': 'U/L',       'normal': (10, 40),     'group': 'liver'},
    'Alkalinephos': {'label': 'Alkaline Phosphatase',    'unit': 'U/L',       'normal': (44, 147),    'group': 'liver'},
    'Bilirubin_direct': {'label': 'Direct Bilirubin',   'unit': 'mg/dL',     'normal': (0, 0.3),     'group': 'liver'},
    'PTT':          {'label': 'PTT (Clotting)',          'unit': 'sec',       'normal': (25, 35),     'group': 'coag'},
    'Fibrinogen':   {'label': 'Fibrinogen',              'unit': 'mg/dL',     'normal': (200, 400),   'group': 'coag'},
    'TroponinI':    {'label': 'Troponin I',              'unit': 'ng/mL',     'normal': (0, 0.04),    'group': 'cardiac'},

    # Demographics
    'Age':          {'label': 'Age',                     'unit': 'years',     'normal': (18, 100),    'group': 'demographic'},
    'Gender':       {'label': 'Gender (0=F, 1=M)',       'unit': '',          'normal': (0, 1),       'group': 'demographic'},
    'ICULOS':       {'label': 'ICU Length of Stay',      'unit': 'hours',     'normal': (0, 720),     'group': 'demographic'},
}

# Critical features — minimum needed for a reliable prediction
CRITICAL_FEATURES = ['HR', 'WBC', 'Lactate', 'Creatinine', 'Platelets', 'Temp', 'Resp', 'SBP']
MINIMUM_FEATURES  = 5   # need at least this many critical features


def extract_from_pdf(pdf_path: str) -> dict:
    """
    Extract lab values from a blood report PDF.
    Returns dict with extracted values and metadata.
    """
    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except ImportError:
        # Fallback: try pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(p.extract_text() or '' for p in pdf.pages)
        except Exception as e:
            return {'error': f'PDF library not available: {e}', 'extracted': {}, 'text': ''}
    except Exception as e:
        return {'error': str(e), 'extracted': {}, 'text': ''}

    extracted = parse_lab_values(text)
    return {
        'extracted':    extracted,
        'text_preview': text[:1000],
        'error':        None
    }


def parse_lab_values(text: str) -> dict:
    """
    Parse numeric lab values from raw text using patterns.
    """
    text_clean = text.replace('\xa0', ' ').replace('\t', ' ')
    results    = {}

    # Pattern map: feature → list of regex aliases to try
    PATTERNS = {
        'HR':        [r'heart\s*rate[:\s]+(\d+\.?\d*)', r'\bHR[:\s]+(\d+\.?\d*)', r'pulse[:\s]+(\d+\.?\d*)'],
        'O2Sat':     [r'(?:spo2|o2\s*sat|oxygen\s*sat)[:\s]+(\d+\.?\d*)', r'\bsao2[:\s]+(\d+\.?\d*)'],
        'Temp':      [r'temp(?:erature)?[:\s]+(\d+\.?\d*)', r'\bT[:\s]+(\d{2}\.?\d*)'],
        'SBP':       [r'systolic[:\s]+(\d+\.?\d*)', r'sbp[:\s]+(\d+\.?\d*)', r'bp[:\s]+(\d+)/\d+'],
        'DBP':       [r'diastolic[:\s]+(\d+\.?\d*)', r'dbp[:\s]+(\d+\.?\d*)', r'bp[:\s]+\d+/(\d+)'],
        'MAP':       [r'map[:\s]+(\d+\.?\d*)', r'mean\s*arterial[:\s]+(\d+\.?\d*)'],
        'Resp':      [r'resp(?:iratory)?[:\s\w]*rate[:\s]+(\d+\.?\d*)', r'\bRR[:\s]+(\d+\.?\d*)'],
        'WBC':       [r'wbc[:\s]+(\d+\.?\d*)', r'white\s*blood[:\s\w]*[:\s]+(\d+\.?\d*)', r'leuko[:\s]+(\d+\.?\d*)'],
        'Lactate':   [r'lactate[:\s]+(\d+\.?\d*)', r'lactic\s*acid[:\s]+(\d+\.?\d*)'],
        'Creatinine':[r'creatinine[:\s]+(\d+\.?\d*)', r'\bcrea(?:t)?[:\s]+(\d+\.?\d*)'],
        'Platelets': [r'platelet[:\s]+(\d+\.?\d*)', r'\bplt[:\s]+(\d+\.?\d*)'],
        'Hgb':       [r'h(?:a?e)?moglobin[:\s]+(\d+\.?\d*)', r'\bhgb[:\s]+(\d+\.?\d*)', r'\bhb[:\s]+(\d+\.?\d*)'],
        'Hct':       [r'hematocrit[:\s]+(\d+\.?\d*)', r'\bhct[:\s]+(\d+\.?\d*)', r'packed\s*cell[:\s]+(\d+\.?\d*)'],
        'pH':        [r'\bph[:\s]+(\d+\.?\d*)', r'arterial\s*ph[:\s]+(\d+\.?\d*)'],
        'PaCO2':     [r'paco2[:\s]+(\d+\.?\d*)', r'co2[:\s]+(\d+\.?\d*)'],
        'HCO3':      [r'hco3[:\s]+(\d+\.?\d*)', r'bicarbonate[:\s]+(\d+\.?\d*)'],
        'Glucose':   [r'glucose[:\s]+(\d+\.?\d*)', r'\bfbs[:\s]+(\d+\.?\d*)', r'blood\s*sugar[:\s]+(\d+\.?\d*)'],
        'BUN':       [r'\bbun[:\s]+(\d+\.?\d*)', r'urea\s*nitrogen[:\s]+(\d+\.?\d*)', r'blood\s*urea[:\s]+(\d+\.?\d*)'],
        'Sodium':    [r'sodium[:\s]+(\d+\.?\d*)', r'\bna[+]?[:\s]+(\d+\.?\d*)'],
        'Potassium': [r'potassium[:\s]+(\d+\.?\d*)', r'\bk[+]?[:\s]+(\d+\.?\d*)'],
        'Chloride':  [r'chloride[:\s]+(\d+\.?\d*)', r'\bcl[-]?[:\s]+(\d+\.?\d*)'],
        'Calcium':   [r'calcium[:\s]+(\d+\.?\d*)', r'\bca[:\s]+(\d+\.?\d*)'],
        'Bilirubin_total': [r'total\s*bili[:\s]+(\d+\.?\d*)', r'bilirubin[,\s]*total[:\s]+(\d+\.?\d*)'],
        'Bilirubin_direct':[r'direct\s*bili[:\s]+(\d+\.?\d*)', r'bilirubin[,\s]*direct[:\s]+(\d+\.?\d*)'],
        'AST':       [r'\bast[:\s]+(\d+\.?\d*)', r'aspartate[:\s]+(\d+\.?\d*)'],
        'Alkalinephos': [r'alkaline\s*phosphatase[:\s]+(\d+\.?\d*)', r'\balp[:\s]+(\d+\.?\d*)'],
        'PTT':       [r'\bptt[:\s]+(\d+\.?\d*)', r'partial\s*thromboplastin[:\s]+(\d+\.?\d*)'],
        'Fibrinogen':[r'fibrinogen[:\s]+(\d+\.?\d*)'],
        'TroponinI': [r'troponin[:\s]+(\d+\.?\d*)', r'tnI[:\s]+(\d+\.?\d*)'],
        'Magnesium': [r'magnesium[:\s]+(\d+\.?\d*)', r'\bmg[:\s]+(\d+\.?\d*)'],
        'Phosphate': [r'phosphate[:\s]+(\d+\.?\d*)', r'\bphos[:\s]+(\d+\.?\d*)'],
        'Age':       [r'age[:\s]+(\d+)', r'(\d+)\s*(?:yr|year)s?\s*(?:old|of\s*age)'],
        'Gender':    [],   # handled separately below
        'ICULOS':    [r'icu\s*(?:los|length)[:\s]+(\d+\.?\d*)'],
        'FiO2':      [r'fio2[:\s]+(\d+\.?\d*)', r'inspired\s*o2[:\s]+(\d+\.?\d*)'],
        'BaseExcess':[r'base\s*excess[:\s]+(-?\d+\.?\d*)'],
    }

    text_lower = text_clean.lower()

    for feature, patterns in PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    val = float(match.group(1))
                    results[feature] = val
                    break
                except ValueError:
                    continue

    # Gender special handling
    if 'Gender' not in results:
        if re.search(r'\b(?:male|m)\b', text_lower) and not re.search(r'\bfe?male\b', text_lower):
            results['Gender'] = 1.0
        elif re.search(r'\bfe?male\b', text_lower):
            results['Gender'] = 0.0

    return results


def assess_completeness(extracted: dict) -> dict:
    """
    Assess how complete the extracted data is for prediction.
    Returns which features are found, missing, and if we have enough.
    """
    found    = {k: v for k, v in extracted.items() if k in ALL_FEATURES}
    missing  = [k for k in ALL_FEATURES if k not in found]
    critical_found   = [f for f in CRITICAL_FEATURES if f in found]
    critical_missing = [f for f in CRITICAL_FEATURES if f not in found]

    completeness_pct = int(len(found) / len(ALL_FEATURES) * 100)
    has_enough       = len(critical_found) >= MINIMUM_FEATURES

    # Classify missing as "important" vs "optional"
    important_missing = critical_missing
    optional_missing  = [f for f in missing if f not in CRITICAL_FEATURES]

    return {
        'total_features':     len(ALL_FEATURES),
        'found_count':        len(found),
        'missing_count':      len(missing),
        'completeness_pct':   completeness_pct,
        'has_enough':         has_enough,
        'critical_found':     critical_found,
        'critical_missing':   critical_missing,
        'important_missing':  important_missing,
        'optional_missing':   optional_missing,
        'found_features':     {k: {'value': v, **ALL_FEATURES[k]} for k, v in found.items()},
        'missing_details':    {k: ALL_FEATURES[k] for k in missing},
    }


def get_abnormal_flags(values: dict) -> list:
    """Flag values outside normal reference ranges."""
    flags = []
    for feat, val in values.items():
        if feat not in ALL_FEATURES or val is None:
            continue
        meta = ALL_FEATURES[feat]
        lo, hi = meta['normal']
        try:
            v = float(val)
            if v < lo:
                flags.append({'feature': feat, 'label': meta['label'], 'value': v,
                               'unit': meta['unit'], 'direction': 'low',
                               'normal': f"{lo}–{hi}"})
            elif v > hi:
                flags.append({'feature': feat, 'label': meta['label'], 'value': v,
                               'unit': meta['unit'], 'direction': 'high',
                               'normal': f"{lo}–{hi}"})
        except (TypeError, ValueError):
            continue
    return flags