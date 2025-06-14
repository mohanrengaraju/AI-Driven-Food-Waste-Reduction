# app/routes/ai.py
from flask import Blueprint, request, jsonify
import requests

ai_bp = Blueprint('ai', __name__)

@ai_bp.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    # AI Model Prediction Logic (Placeholder)
    response = {
        "food_type": "biryani",
        "quantity": "10 plates",
        "category": "non-veg",
        "expiry_risk": "low"
    }
    return jsonify(response), 200
