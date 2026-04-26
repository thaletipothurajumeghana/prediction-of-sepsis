"""
Report route — GET /api/report
Returns summary of predictions and analytics
"""

from flask import Blueprint, jsonify

report_bp = Blueprint('report', __name__)

# Temporary in-memory storage (you can replace with DB later)
PREDICTION_HISTORY = []


def add_prediction(record):
    """
    Call this from predict_route to store predictions
    """
    PREDICTION_HISTORY.append(record)


@report_bp.route('/report', methods=['GET'])
def get_report():
    if not PREDICTION_HISTORY:
        return jsonify({
            "message": "No prediction data available",
            "total_cases": 0,
            "high_risk": 0,
            "low_risk": 0
        })

    total = len(PREDICTION_HISTORY)

    high_risk = sum(1 for r in PREDICTION_HISTORY if r.get("risk_level") in ["HIGH", "VERY HIGH", "CRITICAL"])
    low_risk = total - high_risk

    avg_prob = sum(r.get("probability", 0) for r in PREDICTION_HISTORY) / total

    return jsonify({
        "total_cases": total,
        "high_risk_cases": high_risk,
        "low_risk_cases": low_risk,
        "average_risk_probability": round(avg_prob, 3),
        "recent_predictions": PREDICTION_HISTORY[-5:]
    })