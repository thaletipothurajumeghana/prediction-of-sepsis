"""
Sepsis Prediction Model — Training Pipeline
============================================
Dataset: PhysioNet Challenge 2019 (via Kaggle)
Target : SepsisLabel (0 = No Sepsis, 1 = Sepsis)
Models : Random Forest, XGBoost, Logistic Regression (ensemble)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import json

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = "data/Dataset.csv"   # Update to your downloaded file path
OUTPUT_DIR  = "outputs"
MODELS_DIR  = "models"
RANDOM_SEED = 42
TEST_SIZE   = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────
VITAL_FEATURES = [
    'HR',          # Heart Rate (bpm)
    'O2Sat',       # Oxygen Saturation (%)
    'Temp',        # Temperature (°C)
    'SBP',         # Systolic Blood Pressure (mmHg)
    'MAP',         # Mean Arterial Pressure (mmHg)
    'DBP',         # Diastolic Blood Pressure (mmHg)
    'Resp',        # Respiratory Rate (breaths/min)
    'EtCO2',       # End-tidal CO2 (mmHg)
]

LAB_FEATURES = [
    'BaseExcess',  # Base Excess (mmol/L)
    'HCO3',        # Bicarbonate (mmol/L)
    'FiO2',        # Fraction of Inspired O2
    'pH',          # Arterial Blood pH
    'PaCO2',       # Arterial CO2 Partial Pressure
    'SaO2',        # O2 Saturation from ABG
    'AST',         # Aspartate Aminotransferase (U/L)
    'BUN',         # Blood Urea Nitrogen (mg/dL)
    'Alkalinephos', # Alkaline Phosphatase (U/L)
    'Calcium',     # Calcium (mg/dL)
    'Chloride',    # Chloride (mEq/L)
    'Creatinine',  # Creatinine (mg/dL)
    'Bilirubin_direct', # Direct Bilirubin (mg/dL)
    'Glucose',     # Glucose (mg/dL)
    'Lactate',     # Lactate (mmol/L)
    'Magnesium',   # Magnesium (mEq/L)
    'Phosphate',   # Phosphate (mEq/L)
    'Potassium',   # Potassium (mEq/L)
    'Bilirubin_total',  # Total Bilirubin (mg/dL)
    'TroponinI',   # Troponin I (ng/mL)
    'Hct',         # Hematocrit (%)
    'Hgb',         # Hemoglobin (g/dL)
    'PTT',         # Partial Thromboplastin Time (sec)
    'WBC',         # White Blood Cell count (x10³/µL)
    'Fibrinogen',  # Fibrinogen (mg/dL)
    'Platelets',   # Platelet Count (x10³/µL)
]

DEMOGRAPHIC_FEATURES = [
    'Age',         # Age (years)
    'Gender',      # Gender (0=F, 1=M)
    'Unit1',       # ICU unit type 1
    'Unit2',       # ICU unit type 2
    'HospAdmTime', # Hours between hospital & ICU admit
    'ICULOS',      # ICU length of stay (hours)
]

ALL_FEATURES = VITAL_FEATURES + LAB_FEATURES + DEMOGRAPHIC_FEATURES
TARGET = 'SepsisLabel'


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def load_data(path: str, sample_size: int = 100000) -> pd.DataFrame:
    """
    Load the sepsis dataset.
    Supports both:
      - Single CSV (Kaggle flat format)
      - PSV files from PhysioNet (auto-merged)
    
    Args:
        path: Path to the dataset
        sample_size: If dataset is larger, sample this many rows for faster training
    """
    print(f"\n{'='*55}")
    print("  SEPSIS PREDICTION — TRAINING PIPELINE")
    print(f"{'='*55}\n")

    if path.endswith('.psv'):
        # PhysioNet format: merge all PSV patient files
        import glob
        files = glob.glob(os.path.dirname(path) + '/*.psv')
        dfs = []
        for f in files:
            df = pd.read_csv(f, sep='|')
            df['patient_id'] = os.path.basename(f).replace('.psv', '')
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
        print(f"[✓] Loaded {len(files)} PhysioNet PSV files → {len(data)} rows")
    else:
        data = pd.read_csv(path)
        print(f"[✓] Loaded dataset: {data.shape[0]} rows × {data.shape[1]} columns")
    
    # Sample data if too large (for faster training)
    if len(data) > sample_size:
        print(f"[→] Sampling {sample_size} rows for faster training...")
        data = data.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"[✓] Sampled dataset: {data.shape[0]} rows × {data.shape[1]} columns")
        print(f"[✓] Loaded dataset: {data.shape[0]} rows × {data.shape[1]} columns")

    return data


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> tuple:
    """
    - Select relevant features
    - Handle missing values
    - Encode categoricals
    - Balance classes with SMOTE
    """
    print("\n[→] Preprocessing...")

    # Keep only available columns
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    missing_features   = [f for f in ALL_FEATURES if f not in df.columns]

    if missing_features:
        print(f"    ⚠ Missing features (will skip): {missing_features[:8]}{'...' if len(missing_features)>8 else ''}")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Check your dataset.")

    X = df[available_features].copy()
    y = df[TARGET].copy()

    # Drop rows where target is null
    mask = y.notna()
    X, y = X[mask], y[mask]
    y = y.astype(int)

    print(f"    Features used   : {len(available_features)}")
    print(f"    Total samples   : {len(y)}")
    print(f"    Sepsis cases    : {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"    Non-sepsis cases: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")

    # Missing value rates
    missing_pct = (X.isnull().mean() * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 70].index.tolist()
    if high_missing:
        print(f"    Dropping {len(high_missing)} features with >70% missing values")
        X = X.drop(columns=high_missing)

    # Store feature names after dropping
    feature_names = X.columns.tolist()

    # Impute remaining missing values
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Apply SMOTE to training set only (handles class imbalance)
    print("\n[→] Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"    Before SMOTE: {y_train.sum()} sepsis / {(y_train==0).sum()} non-sepsis")
    print(f"    After  SMOTE: {y_train_res.sum()} sepsis / {(y_train_res==0).sum()} non-sepsis")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled  = scaler.transform(X_test)

    return (X_train_scaled, X_test_scaled, y_train_res, y_test,
            scaler, imputer, feature_names)


# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────
def build_models():
    """Define the model ensemble."""
    lr = LogisticRegression(
        C=0.1, max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED
    )

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=4,
        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=10,   # handles imbalance
        use_label_encoder=False, eval_metric='logloss',
        random_state=RANDOM_SEED, n_jobs=-1
    )

    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('xgb', xgb_model)],
        voting='soft'
    )

    return {
        'Logistic Regression': lr,
        'Random Forest':       rf,
        'XGBoost':             xgb_model,
        'Ensemble (Soft Vote)': ensemble,
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    """Train all models, evaluate, and return the best one."""
    print("\n[→] Training models...\n")
    models = build_models()
    results = {}

    for name, model in models.items():
        print(f"    Training {name}...")
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        f1  = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        ap  = average_precision_score(y_test, y_proba)

        results[name] = {
            'model':   model,
            'y_pred':  y_pred,
            'y_proba': y_proba,
            'roc_auc': round(auc, 4),
            'f1':      round(f1,  4),
            'accuracy':round(acc, 4),
            'avg_precision': round(ap, 4),
        }

        print(f"        ROC-AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}  AP={ap:.4f}")

    # Pick best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    print(f"\n[★] Best model: {best_name} (AUC={results[best_name]['roc_auc']})")

    return results, best_name, models


# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_results(results, best_name, X_test, y_test, feature_names):
    """Generate evaluation plots."""
    print("\n[→] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Sepsis Prediction Model — Evaluation', fontsize=16, fontweight='bold')

    colors = ['#2563eb', '#16a34a', '#dc2626', '#9333ea']

    # ── ROC Curves
    ax = axes[0, 0]
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})",
                color=colors[i % len(colors)], lw=2)
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Precision-Recall Curves
    ax = axes[0, 1]
    for i, (name, res) in enumerate(results.items()):
        prec, rec, _ = precision_recall_curve(y_test, res['y_proba'])
        ax.plot(rec, prec, label=f"{name} (AP={res['avg_precision']:.3f})",
                color=colors[i % len(colors)], lw=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Confusion Matrix (best model)
    ax = axes[1, 0]
    cm = confusion_matrix(y_test, results[best_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Sepsis','Sepsis'],
                yticklabels=['No Sepsis','Sepsis'])
    ax.set_title(f'Confusion Matrix — {best_name}')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    # ── Feature Importance (Random Forest or XGBoost)
    ax = axes[1, 1]
    fi_model = results.get('XGBoost') or results.get('Random Forest')
    if fi_model:
        model_obj = fi_model['model']
        if hasattr(model_obj, 'feature_importances_'):
            imp = pd.Series(model_obj.feature_importances_, index=feature_names)
            top = imp.nlargest(15)
            top.sort_values().plot(kind='barh', ax=ax, color='#2563eb', alpha=0.8)
            ax.set_title('Top 15 Feature Importances')
            ax.set_xlabel('Importance')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'model_evaluation.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved → {out_path}")


# ─────────────────────────────────────────────
# 5. SAVE ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(results, best_name, scaler, imputer, feature_names):
    """Save best model, scaler, imputer, and metadata."""
    best_model = results[best_name]['model']

    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    joblib.dump(scaler,     os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(imputer,    os.path.join(MODELS_DIR, 'imputer.pkl'))

    meta = {
        'best_model_name': best_name,
        'feature_names':   feature_names,
        'metrics': {
            name: {k: v for k, v in res.items() if k not in ('model','y_pred','y_proba')}
            for name, res in results.items()
        }
    }
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[✓] Artifacts saved to '{MODELS_DIR}/'")
    print(f"      best_model.pkl  — {best_name}")
    print(f"      scaler.pkl      — StandardScaler")
    print(f"      imputer.pkl     — MedianImputer")
    print(f"      metadata.json   — metrics & feature list")


# ─────────────────────────────────────────────
# 6. PRINT FINAL REPORT
# ─────────────────────────────────────────────
def print_report(results, best_name, y_test):
    print(f"\n{'='*55}")
    print("  FINAL MODEL COMPARISON")
    print(f"{'='*55}")
    print(f"  {'Model':<25} {'AUC':>7} {'F1':>7} {'Accuracy':>9}")
    print(f"  {'-'*50}")
    for name, res in results.items():
        star = " ★" if name == best_name else ""
        print(f"  {name:<25} {res['roc_auc']:>7.4f} {res['f1']:>7.4f} {res['accuracy']:>9.4f}{star}")

    print(f"\n  Detailed report for best model ({best_name}):")
    print(f"  {'-'*50}")
    print(classification_report(
        y_test, results[best_name]['y_pred'],
        target_names=['No Sepsis', 'Sepsis'],
        digits=4
    ))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Load
    df = load_data(DATA_PATH)

    # Preprocess
    (X_train, X_test, y_train, y_test,
     scaler, imputer, feature_names) = preprocess(df)

    # Train & evaluate
    results, best_name, models = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_names
    )

    # Plots
    plot_results(results, best_name, X_test, y_test, feature_names)

    # Save
    save_artifacts(results, best_name, scaler, imputer, feature_names)

    # Report
    print_report(results, best_name, y_test)

    print(f"\n{'='*55}")
    print("  Training complete! Use predict.py to run inference.")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
