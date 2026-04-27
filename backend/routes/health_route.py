"""Health check route"""
from flask import Blueprint, jsonify
from backend.utils.ml_engine import model_info

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health():
    info = model_info()
    return jsonify({
        'status':  'ok',
        'version': '1.0.0',
        'model':   info,
    })