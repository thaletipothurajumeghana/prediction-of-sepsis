"""
Health check route — GET /api/health
"""

from flask import Blueprint, jsonify

# SAFE IMPORT (important)
try:
    from utils.ml_engine import model_info
except:
    def model_info():
        return {"loaded": False}

# MUST be at top level
health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health():
    info = model_info()
    return jsonify({
        'status':  'ok',
        'version': '1.0.0',
        'model':   info,
    })